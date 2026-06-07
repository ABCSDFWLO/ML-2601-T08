from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from halfweg.boxoban import parse_level_file
from halfweg.models import ActionModel, ModelConfig, StateModel
from halfweg.planner import HalfWegPlanner, PlannerConfig
from halfweg.replay import ReplayBuffer, ReplayItem
from halfweg.sokoban import SokobanLevel
from halfweg.train import TrainConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run HalfWeg planner on a single level")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--level-file", type=Path, required=True)
    p.add_argument("--level-index", type=int, default=0)
    p.add_argument("--player-mode", type=str, default="all", choices=["random", "all"])
    p.add_argument("--searches", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg_dict = ckpt["cfg"]
    cfg = TrainConfig(**cfg_dict)

    model_cfg = ModelConfig(channels=10, filters=cfg.filters, blocks=cfg.blocks, d=cfg.d, action_dim=5)
    ma = ActionModel(model_cfg)
    ms = StateModel(model_cfg)
    ma.load_state_dict(ckpt["ma"])
    ms.load_state_dict(ckpt["ms"])

    specs = parse_level_file(args.level_file)
    spec = specs[args.level_index]
    level = SokobanLevel.from_ascii(spec.rows, level_id=spec.level_id)

    replay = ReplayBuffer(capacity=10000, seed=cfg.seed)
    replay.add(ReplayItem(level_id=level.level_id, state=level.initial_state))

    planner = HalfWegPlanner(
        action_model=ma,
        state_model=ms,
        cfg=PlannerConfig(d=cfg.d, R=cfg.R, ndss=cfg.ndss, device=args.device),
        replay=replay,
        seed=cfg.seed,
    )

    rng = random.Random(cfg.seed)
    targets = level.state_with_boxes_on_goals(rng, player_mode=args.player_mode)
    if not targets:
        print("No valid targets generated")
        return

    solved = False
    best_len = None
    for target in targets:
        for _ in range(max(1, args.searches)):
            plan = planner.search_policy(cfg.R, level, level.initial_state, target, 0)
            end = level.rollout(level.initial_state, plan)
            replay.add(ReplayItem(level_id=level.level_id, state=end))
            if level.solved(end):
                solved = True
                plen = len(plan)
                if best_len is None or plen < best_len:
                    best_len = plen

    print(f"level={level.level_id} solved={solved} best_plan_len={best_len}")


if __name__ == "__main__":
    main()
