import pprint

import numpy as np
import torch
import environments
from environments import env_base
from environments.sokoban.sokoban_env import CHANNEL_BOX, CHANNEL_GOAL, DIRECTIONS
import helpers
from hw_impl import env_torch_wrapper, hw_common, hw_experience_replay, hw_policies, model_mgmt


def get_policy(policy_name: str, model_keeper: model_mgmt.ModelKeeper, long_memory_envs_sampler: hw_experience_replay.MemoryEnvsSampler) -> hw_policies.BasePolicy:
    plta = model_keeper.models['PLTA']

    layers = []
    layers.append(hw_policies.FixedFullScanPolicy(plta))
    if policy_name == 'fixed_full_scan':
        return layers[-1]

    layers.append(hw_policies.PolicyZeroLevel(plta, layers))
    if policy_name == 'PL0':
        return layers[-1]

    if policy_name.startswith('PL'):
        pl_i = int(policy_name.replace('PL', ''))
        for i in range(1, pl_i + 1):
            planner_layer = hw_policies.PolicyHighLevel(model_keeper, long_memory_envs_sampler, layers, layer_i=i + 1)
            layers.append(planner_layer)
        return layers[-1]

    else:
        raise Exception(f"Planner '{policy_name}' not found")


def get_policies(model_keeper: model_mgmt.ModelKeeper, config_policy_name: str) -> list[str]:
    if config_policy_name is None or config_policy_name == 'all':
        model_states = model_keeper.models['PLHW']
        result = ['PL0']
        for i in range(model_states.PLHW_LAYERS):
            result.append(f"PL{i+1}")

    elif config_policy_name == 'last':
        model_states = model_keeper.models['PLHW']
        result = [f"PL{model_states.PLHW_LAYERS}"]

    else:
        result = [config_policy_name]

    return result


def _solve__one_shot(
        device,
        start_envs: env_torch_wrapper.EnvsTensorList,
        targets: list[torch.Tensor],
        policy: hw_policies.BasePolicy,
        towards_or_away: bool) -> tuple[list[list[np.ndarray]], list[dict]]:
    assert len(targets) == len(start_envs)

    plans = []

    for game_i in range(len(targets)):
        curr_targets = targets[game_i]

        target_env = env_torch_wrapper.EnvsTensorList(states_t=curr_targets)

        b = hw_common.get_b_array_from_towards_or_away(towards_or_away, cnt=len(curr_targets), device=device)

        curr_s0_env = env_torch_wrapper.EnvsTensorList(envs=[start_envs.envs[game_i]])

        curr_plans = policy.get_plan_envs_to_envs(s0=curr_s0_env.tile(len(curr_targets)), target=target_env, b=b)
        curr_plans = list(curr_plans.cpu().numpy())

        plans.append(curr_plans)

    game_stats = [{} for _ in range(len(targets))]
    return plans, game_stats


def _compute_state_mismatch(curr_state: np.ndarray, target_state: np.ndarray) -> float:
    # Replanning mismatch is box-centric to avoid overreacting to player micro-position drift.
    if curr_state.shape[0] > 2 and target_state.shape[0] > 2:
        curr_boxes = curr_state[2]
        target_boxes = target_state[2]
        return float(np.mean(np.abs(curr_boxes - target_boxes)))
    return float(np.mean(np.abs(curr_state - target_state)))


def _solve__closed_loop_replan(
        device,
        start_envs: env_torch_wrapper.EnvsTensorList,
        targets: list[torch.Tensor],
        targets_np: list[np.ndarray],
        policy: hw_policies.BasePolicy,
        towards_or_away: bool,
        AS: int,
        n_max_episode_steps: int,
        replan_every_actions: int,
        replan_mismatch_threshold: float,
        replan_stall_steps: int,
        min_commit_steps: int,
        progress_every_targets: int) -> tuple[list[list[np.ndarray]], list[dict]]:
    assert len(targets) == len(start_envs)

    plans = []
    game_stats = []

    for game_i in range(len(targets)):
        print(f">>> [Replan] Planning Game {game_i + 1}/{len(targets)}", flush=True)
        curr_targets_t = targets[game_i]
        curr_targets_np = targets_np[game_i]

        curr_plans = []
        target_replan_counts = []
        target_box_push_counts = []
        target_timed_outs = []

        for target_i in range(len(curr_targets_t)):
            if progress_every_targets > 0 and (target_i % progress_every_targets == 0):
                print(f">>> [Replan] Game {game_i + 1}: target {target_i + 1}/{len(curr_targets_t)}", flush=True)
            target_t = curr_targets_t[target_i:target_i+1]
            target_np = curr_targets_np[target_i]

            curr_env = start_envs.envs[game_i].copy()
            executed_actions = []

            curr_state = curr_env.get_model_input_s()
            prev_box_state = curr_state[2].copy() if curr_state.shape[0] > 2 else None
            steps_without_box_change = 0

            replan_count = 0
            box_push_count = 0

            while len(executed_actions) < n_max_episode_steps and not curr_env.done:
                target_env = env_torch_wrapper.EnvsTensorList(states_t=target_t)
                b = hw_common.get_b_array_from_towards_or_away(towards_or_away, cnt=1, device=device)
                curr_s0_env = env_torch_wrapper.EnvsTensorList(envs=[curr_env])

                plan_t = policy.get_plan_envs_to_envs(s0=curr_s0_env, target=target_env, b=b)
                plan_np = plan_t[0].detach().cpu().numpy()
                replan_count += 1

                actions_taken_this_replan = 0
                has_non_stop_action = False

                for action in plan_np:
                    action = int(action)
                    if action == AS:
                        break

                    has_non_stop_action = True

                    # Detect box push before stepping
                    if 0 <= action < 4:
                        dr, dc = DIRECTIONS[action]
                        nr = curr_env.player_row + dr
                        nc = curr_env.player_col + dc
                        board = curr_env.board
                        if 0 <= nr < board.shape[1] and 0 <= nc < board.shape[2]:
                            if board[CHANNEL_BOX, nr, nc] == 1:
                                box_push_count += 1

                    reward, done = curr_env.step(action)
                    executed_actions.append(action)
                    actions_taken_this_replan += 1

                    if reward == 1 or done or len(executed_actions) >= n_max_episode_steps:
                        break

                    curr_state = curr_env.get_model_input_s()
                    mismatch = _compute_state_mismatch(curr_state, target_np)

                    if prev_box_state is not None:
                        curr_box_state = curr_state[2]
                        if np.array_equal(curr_box_state, prev_box_state):
                            steps_without_box_change += 1
                        else:
                            steps_without_box_change = 0
                            prev_box_state = curr_box_state.copy()

                    # Keep a short commitment window to avoid excessive replan calls.
                    if actions_taken_this_replan < min_commit_steps:
                        continue

                    if mismatch > replan_mismatch_threshold:
                        break

                    if replan_stall_steps > 0 and steps_without_box_change >= replan_stall_steps:
                        break

                    if replan_every_actions > 0 and actions_taken_this_replan >= replan_every_actions:
                        break

                if not has_non_stop_action:
                    break

                if curr_env.done or len(executed_actions) >= n_max_episode_steps:
                    break

            timed_out = (len(executed_actions) >= n_max_episode_steps) and not curr_env.done
            curr_plans.append(np.array(executed_actions, dtype=np.int64))
            target_replan_counts.append(replan_count)
            target_box_push_counts.append(box_push_count)
            target_timed_outs.append(timed_out)

        plans.append(curr_plans)
        game_stats.append({
            'replan_counts': target_replan_counts,
            'box_push_counts': target_box_push_counts,
            'timed_outs': target_timed_outs,
        })

    return plans, game_stats


@torch.no_grad()
def validate_puzzle_solving__impl(
        config: dict,
        method: str,
        device,
        envs_manager: env_base.BaseEnvsManager,
        model_keeper: model_mgmt.ModelKeeper,
        n_games_to_solve: int,
        policies: list[hw_policies.BasePolicy],
        towards_or_away_array: bool,
        tensorboard):
    model_keeper.eval()

    games_to_solve = []
    for game_i in range(n_games_to_solve):
        env_key, start_env = envs_manager.create_env_with_key()
        games_to_solve.append((env_key, start_env, []))

    targets_np = [start_env.get_target_states() for _, start_env, _ in games_to_solve]
    if config['evaluate']['targets'] == 'random':
        for game_i in range(len(targets_np)):
            ti = np.random.randint(len(targets_np[game_i]))
            targets_np[game_i] = targets_np[game_i][ti:ti+1, ...]
    targets_t = [torch.as_tensor(target_np, dtype=torch.float32, device=device) for target_np in targets_np]

    start_envs = env_torch_wrapper.EnvsTensorList(envs=[start_env.copy() for _, start_env, _ in games_to_solve])
    start_envs.to(device)

    AS = model_keeper.models['PLTA'].AS

    all_results = []

    for policy in policies:
        stat_navigation_mse = dict()

        for towards_or_away in towards_or_away_array:

            stat_proposed_plan_lengths = []
            stat_solved_plan_lengths = []
            stat_n_solutions = []
            stat_mse = []
            stat_solved = []
            stat_replan_counts = []
            stat_box_push_counts = []
            stat_timed_out = []
            stat_partial_progress_unsolved = []

            if method == 'one_shot':
                plans, game_stats = _solve__one_shot(device, start_envs, targets_t, policy, towards_or_away)
            elif method == 'closed_loop_replan':
                replan_every_actions = int(config['evaluate'].get('replan_every_actions', 8))
                replan_mismatch_threshold = float(config['evaluate'].get('replan_mismatch_threshold', 0.08))
                replan_stall_steps = int(config['evaluate'].get('replan_stall_steps', 10))
                min_commit_steps = int(config['evaluate'].get('min_commit_steps', 4))
                progress_every_targets = int(config['evaluate'].get('progress_every_targets', 50))
                n_max_episode_steps = int(config['env']['n_max_episode_steps'])

                plans, game_stats = _solve__closed_loop_replan(
                    device=device,
                    start_envs=start_envs,
                    targets=targets_t,
                    targets_np=targets_np,
                    policy=policy,
                    towards_or_away=towards_or_away,
                    AS=AS,
                    n_max_episode_steps=n_max_episode_steps,
                    replan_every_actions=replan_every_actions,
                    replan_mismatch_threshold=replan_mismatch_threshold,
                    replan_stall_steps=replan_stall_steps,
                    min_commit_steps=min_commit_steps,
                    progress_every_targets=progress_every_targets,
                )
            else:
                raise Exception(f"Unknown validation method `{method}`")

            progress_every_games = int(config['evaluate'].get('progress_every_games', 1))
            for game_i in range(len(games_to_solve)):
                if progress_every_games > 0 and (game_i % progress_every_games == 0):
                    print(f">>> [Progress] Processing Game {game_i + 1}/{len(games_to_solve)}", flush=True)
                curr_plans = plans[game_i]
                g_stats = game_stats[game_i]

                stat_n_solutions.append(len(curr_plans))
                solution_plan = None

                for target_i, curr_plan in enumerate(curr_plans):
                    copy_env = start_envs.envs[game_i].copy()
                    reward, done = copy_env.play_plan_1d(curr_plan, AS)

                    stat_proposed_plan_lengths.append(len(curr_plan))
                    stat_mse.append(np.sum(np.abs(np.expand_dims(copy_env.get_model_input_s(), 0) - targets_np[game_i])) / len(targets_np[game_i]))

                    if g_stats.get('replan_counts'):
                        stat_replan_counts.append(g_stats['replan_counts'][target_i] if target_i < len(g_stats['replan_counts']) else 0)
                    if g_stats.get('box_push_counts'):
                        stat_box_push_counts.append(g_stats['box_push_counts'][target_i] if target_i < len(g_stats['box_push_counts']) else 0)
                    if g_stats.get('timed_outs'):
                        stat_timed_out.append(int(g_stats['timed_outs'][target_i]) if target_i < len(g_stats['timed_outs']) else 0)

                    if reward == 1:
                        solution_plan = curr_plan.copy()

                if solution_plan is not None:
                    stat_solved_plan_lengths.append(len(solution_plan))
                    stat_solved.append(1)
                else:
                    stat_solved.append(0)
                    # Partial progress: boxes on goals at final state of best (last) plan
                    if curr_plans:
                        last_env = start_envs.envs[game_i].copy()
                        last_env.play_plan_1d(curr_plans[-1], AS)
                        board = last_env.get_model_input_s()
                        total_boxes = int(np.sum(board[CHANNEL_BOX]))
                        boxes_on_goals = int(np.sum(board[CHANNEL_BOX] * board[CHANNEL_GOAL]))
                        partial = boxes_on_goals / total_boxes if total_boxes > 0 else 0.0
                        stat_partial_progress_unsolved.append(partial)

            stat_navigation_mse[towards_or_away] = np.mean(stat_mse)

            def _pct(arr, q):
                return float(np.percentile(arr, q)) if arr else float('nan')

            validation_result = {
                'method': method,
                'towards_or_away': towards_or_away,
                'policy': str(policy),
                'games_cnt': len(stat_solved),
                'solved_mean': float(np.mean(stat_solved)),
                'solved_count': int(np.sum(stat_solved)),
                'timeout_rate': float(np.mean(stat_timed_out)) if stat_timed_out else 'n/a',
                'mse_mean': float(np.mean(stat_mse)),
                'proposed_plan_length_mean': float(np.mean(stat_proposed_plan_lengths)),
                'solved_plan_length_mean': float(np.mean(stat_solved_plan_lengths)) if stat_solved_plan_lengths else 'None',
                'solved_plan_length_p50': _pct(stat_solved_plan_lengths, 50),
                'solved_plan_length_p90': _pct(stat_solved_plan_lengths, 90),
                'avg_replan_count': float(np.mean(stat_replan_counts)) if stat_replan_counts else 'n/a',
                'avg_box_push_count': float(np.mean(stat_box_push_counts)) if stat_box_push_counts else 'n/a',
                'partial_progress_unsolved_mean': float(np.mean(stat_partial_progress_unsolved)) if stat_partial_progress_unsolved else 'n/a',
            }
            pprint.pprint(validation_result, width=10000, sort_dicts=False)
            all_results.append(validation_result)

            if tensorboard is not None and towards_or_away:
                tensorboard.append_scalar(f"{str(policy)} solved mean", np.mean(stat_solved))

        if tensorboard is not None and True in stat_navigation_mse and False in stat_navigation_mse:
            tensorboard.append_scalar(f"{str(policy)} navigation spread", stat_navigation_mse[False] - stat_navigation_mse[True])

    return all_results


def go_evaluate(config, device):
    usage = helpers.UsageCounter()

    model_keeper = model_mgmt.ModelKeeper(config)
    model_keeper.to(device)
    envs_sampler = hw_experience_replay.MemoryEnvsSampler(model_keeper=model_keeper)
    usage.checkpoint("Checkpoint loaded")

    envs_manager = environments.create_envs_manager(config['env'])
    usage.checkpoint("Envs loaded")

    policies = []
    for policy_name in get_policies(model_keeper, config['evaluate']['policies']):
        planner_layer = get_policy(policy_name, model_keeper, envs_sampler)
        policies.append(planner_layer)
    
    n_games_to_solve = config['evaluate']['n_games_to_solve']
    method = config['evaluate']['method']
    towards_or_away_array = [True, False] if config['evaluate']['towards_or_away'] == 'both' else [False] if config['evaluate']['towards_or_away'] == 'away' else [True]

    usage.checkpoint("Pre-solve")
    validate_puzzle_solving__impl(config, method, device, envs_manager, model_keeper, n_games_to_solve, policies, towards_or_away_array, tensorboard=None)
    usage.checkpoint("Solving")

    usage.print_stats()