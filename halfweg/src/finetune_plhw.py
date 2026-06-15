import argparse
import os
import random

import numpy as np
import torch
import yaml

import environments
from hw_impl import model_mgmt


def _sample_batch(envs_manager, batch_size: int):
    s_list = []
    t_list = []

    for _ in range(batch_size):
        _, env = envs_manager.create_env_with_key()
        s = env.get_model_input_s().astype(np.float32)
        ts = env.get_target_states().astype(np.float32)

        t_idx = np.random.randint(len(ts))
        t = ts[t_idx]

        s_list.append(s)
        t_list.append(t)

    s_np = np.stack(s_list, axis=0)
    t_np = np.stack(t_list, axis=0)
    return s_np, t_np


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PLHW on Boxoban with a simple state-regression objective")
    parser.add_argument("--base-config", default="configs/evaluate_boxoban_solve.yaml")
    parser.add_argument("--levels", default="../boxoban-levels/unfiltered/train")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
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

    running = []
    for step in range(1, args.steps + 1):
        s_np, t_np = _sample_batch(envs_manager, args.batch_size)

        s_t = torch.as_tensor(s_np, dtype=torch.float32, device=device)
        t_t = torch.as_tensor(t_np, dtype=torch.float32, device=device)

        # PLHW predicts in the PLTA normalized encoding space (cropped 8x8 board).
        s_enc = plta.forward_model_board_normalize(s_t)
        t_enc = plta.forward_model_board_normalize(t_t)

        min_max = torch.zeros((args.batch_size, 1), dtype=torch.int64, device=device)
        layer_idx = torch.randint(
            low=0,
            high=plhw.PLHW_LAYERS,
            size=(args.batch_size,),
            device=device,
            dtype=torch.int64,
        )

        pred = plhw.forward_model_hw(s_enc, t_enc, min_max, layer_idx)
        loss = torch.nn.functional.mse_loss(pred, t_enc)

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
