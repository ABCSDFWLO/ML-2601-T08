"""
HalfWeg robust fine-tuner.

- Policy-guided trajectory collection (not random-walk imitation)
- Fallback exploration when policy emits STOP immediately
- Prefer policy-generated action windows for PLTA labels
- Allow PLHW learning from high-progress partial trajectories
- Probe-based early stopping and best-checkpoint selection
"""

import glob
import os
import pprint
import random
import sys
import time

import numpy as np
import torch
import yaml

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import environments
from environments.sokoban.sokoban_env import CHANNEL_BOX, CHANNEL_GOAL
from hw_impl import env_torch_wrapper, evaluate, hw_common, hw_experience_replay, model_mgmt


def _pick_target_tensor(env, device):
    targets_np = env.get_target_states()
    ti = np.random.randint(len(targets_np))
    return torch.as_tensor(targets_np[ti:ti + 1], dtype=torch.float32, device=device)


def _boxes_on_goal_ratio(env) -> float:
    board = env.get_model_input_s()
    total_boxes = int(np.sum(board[CHANNEL_BOX]))
    if total_boxes == 0:
        return 0.0
    boxes_on_goals = int(np.sum(board[CHANNEL_BOX] * board[CHANNEL_GOAL]))
    return float(boxes_on_goals / total_boxes)


def collect_guided_trajectories(
    envs_manager,
    policy,
    device,
    AS: int,
    n_levels: int,
    n_attempts_per_level: int,
    n_max_episode_steps: int,
    min_commit_steps: int,
    min_partial_ratio: float,
    min_partial_actions: int,
):
    """
    Returns three lists of items:
      (states, actions, action_is_policy, solved, partial_ratio)
    """
    solved_traj = []
    partial_traj = []
    all_traj = []

    for li in range(n_levels):
        if (li + 1) % max(1, n_levels // 4) == 0:
            print(f"    [collect] level {li + 1}/{n_levels}", flush=True)

        env0 = envs_manager.create_env()

        for _ in range(n_attempts_per_level):
            env = env0.copy()
            target_t = _pick_target_tensor(env, device)

            states = [env.copy()]
            actions = []
            action_is_policy = []

            while len(actions) < n_max_episode_steps and not env.done:
                s0 = env_torch_wrapper.EnvsTensorList(envs=[env])
                target_env = env_torch_wrapper.EnvsTensorList(states_t=target_t)
                b = hw_common.get_b_array_from_towards_or_away(True, cnt=1, device=device)
                plan = policy.get_plan_envs_to_envs(s0=s0, target=target_env, b=b)[0].detach().cpu().numpy()

                acted = 0
                for a in plan:
                    a = int(a)
                    if a == AS:
                        break

                    reward, done = env.step(a)
                    actions.append(a)
                    action_is_policy.append(1)
                    states.append(env.copy())
                    acted += 1

                    if reward == 1 or done or len(actions) >= n_max_episode_steps:
                        break
                    if acted >= min_commit_steps:
                        break

                if acted == 0:
                    # Policy emitted STOP immediately: bootstrap one random valid action.
                    valid = np.where(np.array(env.get_valid_actions_mask()) == 1)[0]
                    if len(valid) == 0:
                        break
                    a = int(np.random.choice(valid))
                    env.step(a)
                    actions.append(a)
                    action_is_policy.append(0)
                    states.append(env.copy())

            if len(actions) == 0:
                continue

            solved = bool(env.done and _boxes_on_goal_ratio(env) >= 1.0)
            partial_ratio = _boxes_on_goal_ratio(env)
            item = (states, actions, action_is_policy, solved, partial_ratio)
            all_traj.append(item)

            if solved:
                solved_traj.append(item)
            elif partial_ratio >= min_partial_ratio and len(actions) >= min_partial_actions:
                partial_traj.append(item)

    return solved_traj, partial_traj, all_traj


def _extract_plta_samples(trajectories, predict_steps: int, AS: int, device):
    ss_list, ts_list, labels_list = [], [], []
    fallback_windows = []

    for states, acts, is_policy, _solved, _ratio in trajectories:
        n = len(states)
        if n < 2:
            continue

        for i in range(n - 1):
            k = min(predict_steps, n - 1 - i)
            if k < 1:
                continue

            label = np.full(predict_steps, AS, dtype=np.int64)
            label[:k] = acts[i:i + k]
            item = (states[i].get_model_input_s(), states[i + k].get_model_input_s(), label)

            pure_policy = bool(np.all(np.array(is_policy[i:i + k]) == 1))
            if pure_policy:
                ss_list.append(item[0])
                ts_list.append(item[1])
                labels_list.append(item[2])
            else:
                fallback_windows.append(item)

    if len(ss_list) == 0 and len(fallback_windows) > 0:
        for s, t, y in fallback_windows:
            ss_list.append(s)
            ts_list.append(t)
            labels_list.append(y)

    if not ss_list:
        return None, None, None

    ss_t = torch.as_tensor(np.stack(ss_list), dtype=torch.float32, device=device)
    ts_t = torch.as_tensor(np.stack(ts_list), dtype=torch.float32, device=device)
    labels_t = torch.as_tensor(np.stack(labels_list), dtype=torch.long, device=device)
    return ss_t, ts_t, labels_t


def _extract_plhw_samples(
    trajectories,
    plta,
    plhw_layers: int,
    predict_steps: int,
    device,
    min_partial_ratio_for_plhw: float,
):
    ss_raw, ts_raw, mid_raw, layer_idx = [], [], [], []

    for states, _acts, _is_policy, solved, ratio in trajectories:
        if not solved and ratio < min_partial_ratio_for_plhw:
            continue

        n = len(states)
        for li in range(plhw_layers):
            span = predict_steps * (2 ** (li + 1))
            mid = span // 2
            if n <= span:
                continue

            max_start = n - span - 1
            for start_i in range(0, max_start + 1, max(1, max_start // 6)):
                ss_raw.append(states[start_i].get_model_input_s())
                mid_raw.append(states[start_i + mid].get_model_input_s())
                ts_raw.append(states[start_i + span].get_model_input_s())
                layer_idx.append(li)

    if not ss_raw:
        return None, None, None, None

    ss_raw = torch.as_tensor(np.stack(ss_raw), dtype=torch.float32, device=device)
    ts_raw = torch.as_tensor(np.stack(ts_raw), dtype=torch.float32, device=device)
    mid_raw = torch.as_tensor(np.stack(mid_raw), dtype=torch.float32, device=device)
    layer_t = torch.as_tensor(np.array(layer_idx, dtype=np.int64), dtype=torch.long, device=device)

    with torch.no_grad():
        ss_enc = plta.forward_model_board_normalize(ss_raw)
        ts_enc = plta.forward_model_board_normalize(ts_raw)
        mid_enc = plta.forward_model_board_normalize(mid_raw)

    return ss_enc, ts_enc, mid_enc, layer_t


def _anchor_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _l2sp_penalty(model, anchor, coeff: float):
    if coeff <= 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, p in model.named_parameters():
        if name in anchor:
            penalty = penalty + torch.mean((p - anchor[name]) ** 2)
    return coeff * penalty


def train_step_plta(plta, optimizer, ss_t, ts_t, labels_t, batch_size: int, anchor, l2sp_coeff: float):
    plta.train()
    n = ss_t.shape[0]
    perm = torch.randperm(n, device=ss_t.device)
    total_ce = 0.0
    steps = 0

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        if len(idx) < 2:
            break

        b = torch.zeros((len(idx), 1), dtype=torch.long, device=ss_t.device)
        ss_enc = plta.forward_model_board_normalize(ss_t[idx])
        ts_enc = plta.forward_model_board_normalize(ts_t[idx])
        logits = plta.forward_model_target_actions__enc(ss_enc, ts_enc, b)

        bsz, t, c = logits.shape
        ce = torch.nn.functional.cross_entropy(logits.view(bsz * t, c), labels_t[idx].view(bsz * t))
        loss = ce + _l2sp_penalty(plta, anchor, l2sp_coeff)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(plta.parameters(), 1.0)
        optimizer.step()

        total_ce += float(ce.item())
        steps += 1

    return total_ce / steps if steps > 0 else float("nan")


def train_step_plhw(plhw, optimizer, ss_enc, ts_enc, mid_enc, layer_t, batch_size: int, anchor, l2sp_coeff: float):
    plhw.train()
    n = ss_enc.shape[0]
    perm = torch.randperm(n, device=ss_enc.device)
    total_mse = 0.0
    steps = 0

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        if len(idx) < 2:
            break

        b = torch.zeros((len(idx), 1), dtype=torch.long, device=ss_enc.device)
        pred = plhw.forward_model_hw(ss=ss_enc[idx], ts=ts_enc[idx], min_max_01=b, layer_idx=layer_t[idx])
        mse = torch.nn.functional.mse_loss(pred, mid_enc[idx])
        loss = mse + _l2sp_penalty(plhw, anchor, l2sp_coeff)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(plhw.parameters(), 1.0)
        optimizer.step()

        total_mse += float(mse.item())
        steps += 1

    return total_mse / steps if steps > 0 else float("nan")


@torch.no_grad()
def quick_probe(config, model_keeper, device, probe_levels_path, n_games: int, n_steps: int):
    env_cfg = dict(config["env"])
    env_cfg["levels"] = probe_levels_path
    env_cfg["n_max_episode_steps"] = n_steps

    envs_manager = environments.create_envs_manager(env_cfg)
    sampler = hw_experience_replay.MemoryEnvsSampler(model_keeper=model_keeper)
    policies = [
        evaluate.get_policy(pn, model_keeper, sampler)
        for pn in evaluate.get_policies(model_keeper, "last")
    ]

    probe_cfg = {
        "env": env_cfg,
        "evaluate": {
            "targets": "all",
            "replan_every_actions": 0,
            "replan_mismatch_threshold": 0.2,
            "replan_stall_steps": 10,
            "min_commit_steps": 4,
            "progress_every_targets": 0,
            "progress_every_games": 0,
        },
    }

    results = evaluate.validate_puzzle_solving__impl(
        config=probe_cfg,
        method="closed_loop_replan",
        device=device,
        envs_manager=envs_manager,
        model_keeper=model_keeper,
        n_games_to_solve=n_games,
        policies=policies,
        towards_or_away_array=[True],
        tensorboard=None,
    )
    return float(results[0]["solved_mean"]) if results else 0.0


def _pick_level_file(levels_path: str, epoch_i: int):
    if os.path.isfile(levels_path):
        return levels_path
    files = sorted(glob.glob(os.path.join(levels_path, "*.txt")))
    if not files:
        raise RuntimeError(f"No level files found under: {levels_path}")
    return files[(epoch_i - 1) % len(files)]


def train(config, device):
    model_keeper = model_mgmt.ModelKeeper(config)
    model_keeper.to(device)

    plta = model_keeper.models["PLTA"]
    plhw = model_keeper.models["PLHW"]
    opt_plta = model_keeper.optimizers["PLTA"]
    opt_plhw = model_keeper.optimizers["PLHW"]
    AS = plta.AS

    tc = config["train"]
    epochs = int(tc["epochs"])
    levels_path = tc["levels"]
    probe_levels = tc["probe_levels"]

    n_max_episode_steps = int(tc.get("n_max_episode_steps", config["env"].get("n_max_episode_steps", 100)))
    n_levels_per_epoch = int(tc.get("n_levels_per_epoch", 64))
    n_attempts_per_level = int(tc.get("n_attempts_per_level", 4))
    min_commit_steps = int(tc.get("min_commit_steps", 4))
    min_partial_ratio = float(tc.get("min_partial_ratio", 0.10))
    min_partial_actions = int(tc.get("min_partial_actions", 4))
    max_partial_ratio = float(tc.get("max_partial_ratio", 0.30))
    plhw_min_partial_ratio = float(tc.get("plhw_min_partial_ratio", 0.10))

    batch_size = int(tc.get("batch_size", 128))
    l2sp_plta = float(tc.get("l2sp_plta", 1e-4))
    l2sp_plhw = float(tc.get("l2sp_plhw", 1e-4))

    probe_every = int(tc.get("probe_every_epochs", 5))
    probe_games = int(tc.get("probe_games", 20))
    probe_steps = int(tc.get("probe_max_steps", 100))
    early_patience = int(tc.get("early_stop_patience", 4))
    min_probe_improve = float(tc.get("min_probe_improve", 1e-3))

    save_every = int(tc.get("save_every_epochs", 20))
    ckpt_dir = tc.get("checkpoints_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    anchor_plta = _anchor_state(plta)
    anchor_plhw = _anchor_state(plhw)

    sampler = hw_experience_replay.MemoryEnvsSampler(model_keeper=model_keeper)
    collector_policy_name = tc.get("collector_policy", "PL0")
    policy = evaluate.get_policy(collector_policy_name, model_keeper, sampler)

    best_probe = -1.0
    no_improve = 0

    print(f"Training levels root/file: {levels_path}")
    print(f"Probe levels: {probe_levels}")
    print(f"Collector policy: {collector_policy_name}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        level_file = _pick_level_file(levels_path, epoch)
        env_cfg = dict(config["env"])
        env_cfg["levels"] = level_file
        env_cfg["n_max_episode_steps"] = n_max_episode_steps
        envs_manager = environments.create_envs_manager(env_cfg)

        solved_traj, partial_traj, all_traj = collect_guided_trajectories(
            envs_manager=envs_manager,
            policy=policy,
            device=device,
            AS=AS,
            n_levels=n_levels_per_epoch,
            n_attempts_per_level=n_attempts_per_level,
            n_max_episode_steps=n_max_episode_steps,
            min_commit_steps=min_commit_steps,
            min_partial_ratio=min_partial_ratio,
            min_partial_actions=min_partial_actions,
        )

        partial_keep = max(8, int(len(solved_traj) * max_partial_ratio))
        random.shuffle(partial_traj)
        train_traj = solved_traj + partial_traj[:partial_keep]

        if not train_traj and all_traj:
            candidates = [x for x in all_traj if len(x[1]) >= max(1, min_partial_actions)]
            if not candidates:
                candidates = all_traj
            candidates = sorted(candidates, key=lambda x: (x[4], len(x[1])), reverse=True)
            train_traj = candidates[: min(len(candidates), max(16, n_levels_per_epoch // 2))]

        random.shuffle(train_traj)

        if not train_traj:
            print(f"[Epoch {epoch:4d}/{epochs}] no usable trajectories (solved=0, partial=0, all=0), skip", flush=True)
            continue

        ss_t, ts_t, labels_t = _extract_plta_samples(train_traj, plta.PREDICT_STEPS, AS, device)
        ss_enc, ts_enc, mid_enc, layer_t = _extract_plhw_samples(
            train_traj,
            plta,
            plhw.PLHW_LAYERS,
            plta.PREDICT_STEPS,
            device,
            plhw_min_partial_ratio,
        )

        plta_loss = float("nan")
        plhw_loss = float("nan")

        if ss_t is not None:
            plta_loss = train_step_plta(plta, opt_plta, ss_t, ts_t, labels_t, batch_size, anchor_plta, l2sp_plta)
        if ss_enc is not None:
            plhw_loss = train_step_plhw(plhw, opt_plhw, ss_enc, ts_enc, mid_enc, layer_t, batch_size, anchor_plhw, l2sp_plhw)

        model_keeper.iter_i += 1
        elapsed = time.time() - t0

        solved_ratio = len(solved_traj) / max(1, len(solved_traj) + len(partial_traj))
        avg_len = float(np.mean([len(x[1]) for x in train_traj])) if train_traj else 0.0

        print(
            f"[Epoch {epoch:4d}/{epochs}] file={os.path.basename(level_file)} "
            f"solved={len(solved_traj)} partial={len(partial_traj)} all={len(all_traj)} used={len(train_traj)} "
            f"solved_ratio={solved_ratio:.3f} avg_len={avg_len:.1f} "
            f"| plta_loss={plta_loss:.4f} plhw_loss={plhw_loss:.4f} | {elapsed:.1f}s",
            flush=True,
        )

        if save_every > 0 and epoch % save_every == 0:
            model_keeper.save_checkpoint(ckpt_dir, f"[Epoch {epoch}]")

        if probe_every > 0 and epoch % probe_every == 0:
            probe = quick_probe(config, model_keeper, device, probe_levels, probe_games, probe_steps)
            print(f"  >>> Probe solve_rate={probe:.3f} (n={probe_games})", flush=True)

            if probe > best_probe + min_probe_improve:
                best_probe = probe
                no_improve = 0
                model_keeper.save_checkpoint(ckpt_dir, f"[Best {probe:.3f}]")
            else:
                no_improve += 1
                print(f"  >>> No probe improvement ({no_improve}/{early_patience})", flush=True)
                if no_improve >= early_patience:
                    print("  >>> Early stop triggered.", flush=True)
                    break

    model_keeper.save_checkpoint(ckpt_dir, "[Final]")
    print(f"Training complete. Best probe solve_rate={best_probe:.3f}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python -u src/train_halfweg.py configs/train_finetune_medium.yaml")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)

    torch.set_num_threads(1)
    torch.autograd.set_detect_anomaly(False)

    if config["infra"].get("device") in (None, "cpu"):
        device = "cpu"
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    pprint.pprint(config)
    train(config, device)


if __name__ == "__main__":
    main()
