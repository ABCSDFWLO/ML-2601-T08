"""
HalfWeg self-play fine-tuner.

Continues training PLTA + PLHW from an existing checkpoint using random-walk
trajectories collected on Boxoban levels.

Training signals:
  PLTA  - CrossEntropy(predicted_actions, actual_actions)
          Label: exact action sequence from random-walk sub-trajectory of length PREDICT_STEPS
          Input: (s_i, s_{i+k}, b=0)

  PLHW  - MSE(predicted_midpoint_enc, actual_midpoint_enc)
          Label: encoded midpoint state at step span/2
          Input: (enc(s_start), enc(s_end), b=0, layer_idx=L)
          At layer L the span = PREDICT_STEPS * 2^(L+1), mid = span/2

Usage (inside container):
    python -u src/train_halfweg.py configs/train_finetune_medium.yaml
"""

import os
import sys
import pprint
import random
import time
from collections import defaultdict

import numpy as np
import torch
import yaml

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import environments
import helpers
from hw_impl import hw_experience_replay, model_mgmt


# ---------------------------------------------------------------------------
# Random-walk trajectory collection
# ---------------------------------------------------------------------------

def _collect_random_walk(env, max_steps: int) -> tuple[list, list]:
    """
    Return (states, actions) along a random walk.
    states[i+1] is the result of actions[i] from states[i].
    Stops early if the environment terminates.
    """
    states = [env.copy()]
    actions = []
    for _ in range(max_steps):
        if env.done:
            break
        mask = env.get_valid_actions_mask()
        valid = np.where(np.array(mask) == 1)[0]
        if len(valid) == 0:
            break
        action = int(np.random.choice(valid))
        env.step(action)
        states.append(env.copy())
        actions.append(action)
    return states, actions


def collect_trajectories(
        envs_manager,
        n_levels: int,
        walk_steps: int,
        n_walks_per_level: int) -> list[tuple[list, list]]:
    """
    Returns flat list of (states, actions) tuples.
    """
    trajectories = []
    for _ in range(n_levels):
        env = envs_manager.create_env()
        for _ in range(n_walks_per_level):
            states, actions = _collect_random_walk(env.copy(), walk_steps)
            if len(states) >= 2:
                trajectories.append((states, actions))
    return trajectories


# ---------------------------------------------------------------------------
# Training sample extraction
# ---------------------------------------------------------------------------

def _extract_plta_samples(trajectories: list[tuple], plta, predict_steps: int, AS: int, device):
    """
    From each (states, actions) trajectory, extract (s_i, s_{i+k}, actions[i:i+k])
    for k = min(predict_steps, available steps from i).
    Actions are already recorded during collection — O(N) extraction.
    """
    ss_list, ts_list, labels_list = [], [], []

    for states, acts in trajectories:
        n = len(states)
        if n < 2:
            continue
        for i in range(n - 1):
            k = min(predict_steps, n - 1 - i)
            if k < 1:
                continue
            label = np.full(predict_steps, AS, dtype=np.int64)
            label[:k] = acts[i:i + k]
            ss_list.append(states[i].get_model_input_s())
            ts_list.append(states[i + k].get_model_input_s())
            labels_list.append(label)

    if not ss_list:
        return None, None, None

    ss_t = torch.as_tensor(np.stack(ss_list), dtype=torch.float32, device=device)
    ts_t = torch.as_tensor(np.stack(ts_list), dtype=torch.float32, device=device)
    labels_t = torch.as_tensor(np.stack(labels_list), dtype=torch.long, device=device)
    return ss_t, ts_t, labels_t


def _extract_plhw_samples(trajectories: list[tuple], plta, plhw_layers: int, predict_steps: int, device):
    """
    For each (states, actions) trajectory, extract PLHW training samples.
    At layer L: span = predict_steps * 2^(L+1), mid = span // 2
    Returns (ss_enc, ts_enc, mid_enc, layer_idx_t).
    """
    result_ss, result_ts, result_mid, result_layer = [], [], [], []

    for states, _acts in trajectories:
        n = len(states)
        for L in range(plhw_layers):
            span = predict_steps * (2 ** (L + 1))
            mid = span // 2
            if n <= span:
                continue
            # Sample a few triplets from this trajectory
            max_start = n - span - 1
            for start_i in range(0, max_start + 1, max(1, max_start // 4)):
                s_start = states[start_i].get_model_input_s()
                s_mid = states[start_i + mid].get_model_input_s()
                s_end = states[start_i + span].get_model_input_s()
                result_ss.append(s_start)
                result_ts.append(s_end)
                result_mid.append(s_mid)
                result_layer.append(L)

    if not result_ss:
        return None, None, None, None

    # Convert to tensors and normalize through PLTA
    ss_raw = torch.as_tensor(np.stack(result_ss), dtype=torch.float32, device=device)
    ts_raw = torch.as_tensor(np.stack(result_ts), dtype=torch.float32, device=device)
    mid_raw = torch.as_tensor(np.stack(result_mid), dtype=torch.float32, device=device)
    layer_t = torch.as_tensor(np.array(result_layer, dtype=np.int64), dtype=torch.long, device=device)

    with torch.no_grad():
        ss_enc = plta.forward_model_board_normalize(ss_raw)
        ts_enc = plta.forward_model_board_normalize(ts_raw)
        mid_enc = plta.forward_model_board_normalize(mid_raw)

    return ss_enc, ts_enc, mid_enc, layer_t


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step_plta(plta, optimizer, ss_t, ts_t, labels_t, batch_size: int, AS: int):
    """
    CrossEntropy loss over predicted action sequences.
    labels_t: (N, PREDICT_STEPS) with values in [0, AS] where AS = stop.
    """
    plta.train()
    N = ss_t.shape[0]
    total_loss = 0.0
    n_batches = 0
    perm = torch.randperm(N, device=ss_t.device)

    b_zero = torch.zeros(batch_size, 1, dtype=torch.long, device=ss_t.device)

    for start in range(0, N, batch_size):
        idx = perm[start:start + batch_size]
        if len(idx) < 2:
            break
        b = b_zero[:len(idx)]

        ss_norm = plta.forward_model_board_normalize(ss_t[idx])
        ts_norm = plta.forward_model_board_normalize(ts_t[idx])

        logits = plta.forward_model_target_actions__enc(ss_norm, ts_norm, b)
        # logits: (B, PREDICT_STEPS, AS+1)
        B, T, C = logits.shape
        loss = torch.nn.functional.cross_entropy(
            logits.view(B * T, C),
            labels_t[idx].view(B * T)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(plta.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else float('nan')


def train_step_plhw(plhw, optimizer, ss_enc, ts_enc, mid_enc, layer_t, batch_size: int):
    """
    MSE loss between predicted midpoint encoding and actual midpoint encoding.
    """
    plhw.train()
    N = ss_enc.shape[0]
    total_loss = 0.0
    n_batches = 0
    perm = torch.randperm(N, device=ss_enc.device)

    b_zero = torch.zeros(batch_size, 1, dtype=torch.long, device=ss_enc.device)

    for start in range(0, N, batch_size):
        idx = perm[start:start + batch_size]
        if len(idx) < 2:
            break
        b = b_zero[:len(idx)]

        pred_mid = plhw.forward_model_hw(
            ss=ss_enc[idx],
            ts=ts_enc[idx],
            min_max_01=b,
            layer_idx=layer_t[idx],
        )
        target_mid = mid_enc[idx].detach()
        loss = torch.nn.functional.mse_loss(pred_mid, target_mid)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(plhw.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else float('nan')


# ---------------------------------------------------------------------------
# Quick solve-rate probe
# ---------------------------------------------------------------------------

@torch.no_grad()
def quick_solve_probe(model_keeper, envs_manager, n_games: int, n_max_steps: int, device) -> float:
    """
    Quick greedy probe using the last policy layer. Returns solve rate.
    """
    from hw_impl import evaluate, env_torch_wrapper, hw_common

    model_keeper.eval()

    envs_sampler = hw_experience_replay.MemoryEnvsSampler(model_keeper=model_keeper)

    policy_names = evaluate.get_policies(model_keeper, 'last')
    policies = [evaluate.get_policy(n, model_keeper, envs_sampler) for n in policy_names]

    config = {
        'env': {'n_max_episode_steps': n_max_steps},
        'evaluate': {
            'targets': 'random',
            'replan_every_actions': 0,
            'replan_mismatch_threshold': 0.20,
            'replan_stall_steps': 10,
            'min_commit_steps': 4,
            'progress_every_targets': 0,
            'progress_every_games': 0,
        }
    }

    results = evaluate.validate_puzzle_solving__impl(
        config=config,
        method='closed_loop_replan',
        device=device,
        envs_manager=envs_manager,
        model_keeper=model_keeper,
        n_games_to_solve=n_games,
        policies=policies,
        towards_or_away_array=[True],
        tensorboard=None,
    )
    if results:
        return float(results[0].get('solved_mean', 0.0))
    return 0.0


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: dict, device):
    model_keeper = model_mgmt.ModelKeeper(config)
    model_keeper.to(device)
    print('Checkpoint loaded')

    plta = model_keeper.models['PLTA']
    plhw = model_keeper.models['PLHW']
    opt_plta = model_keeper.optimizers['PLTA']
    opt_plhw = model_keeper.optimizers['PLHW']

    AS = plta.AS
    PREDICT_STEPS = plta.PREDICT_STEPS
    PLHW_LAYERS = plhw.PLHW_LAYERS

    tc = config['train']
    n_epochs = int(tc['epochs'])
    levels_root = tc['levels']          # may be dir or single file
    walk_steps = int(tc.get('walk_steps', 128))
    n_levels_per_epoch = int(tc.get('n_levels_per_epoch', 32))
    n_walks_per_level = int(tc.get('n_walks_per_level', 4))
    batch_size = int(tc.get('batch_size', 128))
    probe_every = int(tc.get('probe_every_epochs', 5))
    probe_games = int(tc.get('probe_games', 20))
    probe_steps = int(tc.get('probe_max_steps', 100))
    save_every = int(tc.get('save_every_epochs', 10))
    checkpoints_dir = tc.get('checkpoints_dir', 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Collect the list of level files to rotate through
    if os.path.isfile(levels_root):
        level_files = [levels_root]
    else:
        import glob as _glob
        level_files = sorted(_glob.glob(os.path.join(levels_root, '*.txt')))
    if not level_files:
        raise RuntimeError(f"No level files found under: {levels_root}")
    print(f'Training level files pool: {len(level_files)} file(s)')

    # Probe env manager loaded once from a fixed file
    probe_levels_path = tc.get('probe_levels', level_files[0])
    if os.path.isfile(probe_levels_path):
        probe_file = probe_levels_path
    else:
        import glob as _glob2
        probe_files = sorted(_glob2.glob(os.path.join(probe_levels_path, '*.txt')))
        probe_file = probe_files[0] if probe_files else level_files[0]
    probe_env_config = dict(config['env'])
    probe_env_config['levels'] = probe_file
    probe_envs_manager = environments.create_envs_manager(probe_env_config)
    print(f'Probe levels: {probe_file}')

    # Cache a loaded envs_manager per file to avoid re-parsing every epoch
    _envs_cache: dict[str, object] = {}

    def _get_envs_manager(filepath: str):
        if filepath not in _envs_cache:
            cfg = dict(config['env'])
            cfg['levels'] = filepath
            _envs_cache[filepath] = environments.create_envs_manager(cfg)
        return _envs_cache[filepath]

    best_solve_rate = -1.0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # Rotate through files (lazy-load and cache)
        file_idx = (epoch - 1) % len(level_files)
        cur_file = level_files[file_idx]
        envs_manager = _get_envs_manager(cur_file)

        # --- Collect trajectories ---
        trajectories = collect_trajectories(
            envs_manager=envs_manager,
            n_levels=n_levels_per_epoch,
            walk_steps=walk_steps,
            n_walks_per_level=n_walks_per_level,
        )
        n_traj = len(trajectories)
        avg_len = np.mean([len(t) for t in trajectories]) if trajectories else 0

        # --- PLTA training ---
        ss_t, ts_t, labels_t = _extract_plta_samples(trajectories, plta, PREDICT_STEPS, AS, device)
        if ss_t is not None:
            plta_loss = train_step_plta(plta, opt_plta, ss_t, ts_t, labels_t, batch_size, AS)
        else:
            plta_loss = float('nan')

        # --- PLHW training ---
        ss_enc, ts_enc, mid_enc, layer_t = _extract_plhw_samples(trajectories, plta, PLHW_LAYERS, PREDICT_STEPS, device)
        if ss_enc is not None:
            plhw_loss = train_step_plhw(plhw, opt_plhw, ss_enc, ts_enc, mid_enc, layer_t, batch_size)
        else:
            plhw_loss = float('nan')

        elapsed = time.time() - t0
        model_keeper.iter_i += 1

        print(
            f'[Epoch {epoch:4d}/{n_epochs}] '
            f'traj={n_traj} avg_len={avg_len:.1f} | '
            f'plta_loss={plta_loss:.4f} plhw_loss={plhw_loss:.4f} | '
            f'{elapsed:.1f}s',
            flush=True,
        )

        # --- Probe solve rate ---
        if probe_every > 0 and epoch % probe_every == 0:
            solve_rate = quick_solve_probe(model_keeper, probe_envs_manager, probe_games, probe_steps, device)
            print(f'  >>> Probe solve_rate={solve_rate:.3f} (n={probe_games})', flush=True)

            if solve_rate > best_solve_rate:
                best_solve_rate = solve_rate
                model_keeper.save_checkpoint(checkpoints_dir, f'[Best {solve_rate:.3f}]')

        # --- Periodic save ---
        if save_every > 0 and epoch % save_every == 0:
            model_keeper.save_checkpoint(checkpoints_dir, f'[Epoch {epoch}]')

    # Final save
    model_keeper.save_checkpoint(checkpoints_dir, '[Final]')
    print(f'Training complete. Best probe solve_rate={best_solve_rate:.3f}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print('Usage: python -u src/train_halfweg.py configs/train_finetune_medium.yaml')
        sys.exit(1)

    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)

    torch.set_num_threads(1)
    torch.autograd.set_detect_anomaly(False)

    if config['infra'].get('device') in (None, 'cpu'):
        device = 'cpu'
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'Device: {device}')
    pprint.pprint(config)

    train(config, device)


if __name__ == '__main__':
    main()
