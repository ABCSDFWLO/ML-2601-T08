import argparse
import os
import random

import numpy as np
import torch
import yaml

import environments
from hw_impl import env_torch_wrapper, evaluate as evaluate_impl, hw_common, hw_experience_replay, model_mgmt


def _select_layer_idx_from_plan_length(plan_length: int, layer_count: int) -> int:
    if plan_length <= 0:
        return 0
    layer_idx = int(np.floor(np.log2(plan_length)) - 2)
    return int(np.clip(layer_idx, 0, layer_count - 1))


def _collect_midpoint_sample(env, policy, plta, device, target_sample_count: int):
    start_envs = env_torch_wrapper.EnvsTensorList(envs=[env.copy()])
    target_states = env.get_target_states().astype(np.float32)
    if len(target_states) > target_sample_count:
        chosen = np.random.choice(len(target_states), size=target_sample_count, replace=False)
        target_states = target_states[chosen]
    target_envs = env_torch_wrapper.EnvsTensorList(states_np=target_states)

    b = hw_common.get_b_array_from_towards_or_away(True, cnt=len(target_states), device=device)
    plans_t = policy.get_plan_envs_to_envs(s0=start_envs.tile(len(target_states)), target=target_envs, b=b)
    plans_np = plans_t.cpu().numpy().astype(np.int64)

    best_item = None
    best_score = None

    for plan in plans_np:
        stop_positions = np.where(plan == plta.AS)[0]
        plan_length = int(stop_positions[0]) if len(stop_positions) > 0 else int(len(plan))
        if plan_length < 2:
            continue

        midpoint_i = max(1, plan_length // 2)
        states = start_envs.get_states_on_trajectory(0, plan, [0, midpoint_i, plan_length])
        if len(states) != 3:
            continue

        s_np, m_np, t_np = states
        test_env = env.copy()
        reward, _ = test_env.play_plan_1d(plan, plta.AS)
        final_state = test_env.get_model_input_s().astype(np.float32)
        target_distance = float(np.mean(np.abs(final_state - t_np.astype(np.float32))))
        score = (0 if reward == 1 else 1, target_distance, plan_length)

        if best_score is None or score < best_score:
            layer_idx = _select_layer_idx_from_plan_length(plan_length, policy.model_keeper.models["PLHW"].PLHW_LAYERS)
            best_item = (s_np.astype(np.float32), m_np.astype(np.float32), t_np.astype(np.float32), layer_idx)
            best_score = score

    return best_item


def _sample_batch(envs_manager, policy, plta, batch_size: int, device, target_sample_count: int):
    s_list = []
    m_list = []
    t_list = []
    layer_list = []

    max_attempts = max(batch_size * 20, 20)
    attempts = 0

    while len(s_list) < batch_size and attempts < max_attempts:
        attempts += 1
        _, env = envs_manager.create_env_with_key()
        sample = _collect_midpoint_sample(env, policy, plta, device, target_sample_count)
        if sample is None:
            continue

        s, m, t, layer_idx = sample
        s_list.append(s)
        m_list.append(m)
        t_list.append(t)
        layer_list.append(layer_idx)

    if len(s_list) == 0:
        raise RuntimeError(f"Could not build any midpoint samples after {attempts} attempts for levels={envs_manager}")

    s_np = np.stack(s_list, axis=0)
    m_np = np.stack(m_list, axis=0)
    t_np = np.stack(t_list, axis=0)
    layer_np = np.asarray(layer_list, dtype=np.int64)
    return s_np, m_np, t_np, layer_np


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PLHW on Boxoban with a simple state-regression objective")
    parser.add_argument("--base-config", default="configs/evaluate_boxoban_solve.yaml")
    parser.add_argument("--levels", default="../boxoban-levels/unfiltered/train")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-sample-count", type=int, default=1)
    parser.add_argument("--save", default="trained_models/boxoban_vast_v4_finetuned_gpu.ckpt")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.base_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this script with GPU enabled.")

    device = torch.device("cuda:0")

    config["env"]["levels"] = args.levels
    model_keeper = model_mgmt.ModelKeeper(config)
    model_keeper.to(device)

    plta = model_keeper.models["PLTA"]
    plhw = model_keeper.models["PLHW"]
    plhw.train()
    plta.eval()

    optimizer = torch.optim.AdamW(plhw.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    envs_manager = environments.create_envs_manager(config["env"])
    long_memory_sampler = hw_experience_replay.MemoryEnvsSampler(model_keeper=model_keeper)
    policy = evaluate_impl.get_policy(f"PL{plhw.PLHW_LAYERS}", model_keeper, long_memory_sampler)

    running = []
    for step in range(1, args.steps + 1):
        s_np, m_np, t_np, layer_np = _sample_batch(envs_manager, policy, plta, args.batch_size, device, args.target_sample_count)

        s_t = torch.as_tensor(s_np, dtype=torch.float32, device=device)
        m_t = torch.as_tensor(m_np, dtype=torch.float32, device=device)
        t_t = torch.as_tensor(t_np, dtype=torch.float32, device=device)

        # PLHW predicts in the PLTA normalized encoding space (cropped 8x8 board).
        s_enc = plta.forward_model_board_normalize(s_t)
        m_enc = plta.forward_model_board_normalize(m_t)
        t_enc = plta.forward_model_board_normalize(t_t)

        min_max = torch.zeros((args.batch_size, 1), dtype=torch.int64, device=device)
        layer_idx = torch.as_tensor(layer_np, dtype=torch.int64, device=device)

        pred = plhw.forward_model_hw(s_enc, t_enc, min_max, layer_idx)
        loss = torch.nn.functional.mse_loss(pred, m_enc)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running.append(float(loss.detach().cpu().item()))
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            window = running[-args.log_every:]
            print(f"step={step:04d} loss={running[-1]:.6f} mean_last={np.mean(window):.6f}")

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    data = {
        "models": {name: model.state_dict() for name, model in model_keeper.models.items()},
        "optimizers": {},
        "long_memory": model_keeper.long_memory,
        "iter_i": model_keeper.iter_i,
        "layer_targets": model_keeper.layer_targets,
    }
    torch.save(data, args.save)

    print(f"saved={args.save}")


if __name__ == "__main__":
    main()
