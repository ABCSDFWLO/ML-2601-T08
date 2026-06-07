from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from halfweg.train import HalfWegTrainer, TrainConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a HalfWeg-style hierarchical planner on Boxoban levels")
    p.add_argument("--dataset-root", type=Path, default=Path("boxoban-levels"))
    p.add_argument("--split-glob", type=str, default="unfiltered/train/*.txt")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--d", type=int, default=4)
    p.add_argument("--R", type=int, default=5)
    p.add_argument("--ndss", type=int, default=32)
    p.add_argument("--max-files", type=int, default=4)
    p.add_argument("--max-levels-per-file", type=int, default=128)
    p.add_argument("--planning-samples", type=int, default=128)
    p.add_argument("--self-play-levels", type=int, default=64)
    p.add_argument("--self-play-steps", type=int, default=64)
    p.add_argument("--save", type=Path, default=Path("checkpoints/halfweg.pt"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        epochs=args.epochs,
        device=args.device,
        d=args.d,
        R=args.R,
        ndss=args.ndss,
        max_files=args.max_files,
        max_levels_per_file=args.max_levels_per_file,
        planning_samples=args.planning_samples,
        self_play_levels=args.self_play_levels,
        self_play_steps=args.self_play_steps,
    )

    trainer = HalfWegTrainer(root_path=str(args.dataset_root), split_glob=args.split_glob, cfg=cfg)
    trainer.fit()

    args.save.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(args.save))
    print(f"saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
