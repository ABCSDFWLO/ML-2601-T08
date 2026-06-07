from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from halfweg.boxoban import BoxobanCorpus
from halfweg.models import ActionModel, ModelConfig, StateModel
from halfweg.planner import HalfWegPlanner, PlannerConfig
from halfweg.replay import ReplayBuffer, ReplayItem
from halfweg.train import TrainConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate HalfWeg checkpoint on a Boxoban split")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=Path("boxoban-levels"))
    p.add_argument("--split-glob", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-files", type=int, default=2)
    p.add_argument("--max-levels-per-file", type=int, default=32)
    p.add_argument("--eval-levels", type=int, default=30)
    p.add_argument("--player-mode", type=str, default="all", choices=["random", "all"])
    p.add_argument("--searches", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = TrainConfig(**ckpt["cfg"])

    model_cfg = ModelConfig(channels=10, filters=cfg.filters, blocks=cfg.blocks, d=cfg.d, action_dim=5)
    ma = ActionModel(model_cfg)
    ms = StateModel(model_cfg)
    ma.load_state_dict(ckpt["ma"])
    ms.load_state_dict(ckpt["ms"])

    corpus = BoxobanCorpus(root=args.dataset_root, split_glob=args.split_glob, seed=args.seed)
    corpus.load(max_files=args.max_files, max_levels_per_file=args.max_levels_per_file)
    levels = list(corpus.iter_sokoban_levels())

    replay = ReplayBuffer(capacity=200000, seed=args.seed)
    for lv in levels:
        replay.add(ReplayItem(level_id=lv.level_id, state=lv.initial_state))

    planner = HalfWegPlanner(
        action_model=ma,
        state_model=ms,
        cfg=PlannerConfig(d=cfg.d, R=cfg.R, ndss=cfg.ndss, device=args.device),
        replay=replay,
        seed=args.seed,
    )

    solved = 0
    total = 0
    solved_plan_lengths = []

    for i in range(min(args.eval_levels, len(levels))):
        level = levels[i]
        targets = level.state_with_boxes_on_goals(rng, player_mode=args.player_mode)
        if not targets:
            continue

        ok = False
        best_len = None
        for target in targets:
            for _ in range(max(1, args.searches)):
                plan = planner.search_policy(cfg.R, level, level.initial_state, target, 0)
                end = level.rollout(level.initial_state, plan)
                replay.add(ReplayItem(level_id=level.level_id, state=end))
                if level.solved(end):
                    ok = True
                    plen = len(plan)
                    if best_len is None or plen < best_len:
                        best_len = plen

        total += 1
        if ok:
            solved += 1
            if best_len is not None:
                solved_plan_lengths.append(best_len)

    solve_rate = (solved / total) if total > 0 else 0.0
    avg_len = (sum(solved_plan_lengths) / len(solved_plan_lengths)) if solved_plan_lengths else 0.0

    print(f"split={args.split_glob}")
    print(f"levels={total}")
    print(f"solved={solved}")
    print(f"solve_rate={solve_rate:.4f}")
    print(f"avg_solution_len={avg_len:.2f}")


if __name__ == "__main__":
    main()
