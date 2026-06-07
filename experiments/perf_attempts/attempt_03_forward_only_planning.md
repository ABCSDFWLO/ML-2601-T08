# Attempt 03 - Forward-Only Planning

## Goal
Remove unsupported backward-style planning branches that were sending the planner away from target states.

## Hypothesis
The implementation does not support true reverse Sokoban dynamics, but it still sampled `b=1` branches during self-play, planning-problem generation, and intermediate search legs. Those branches optimized for larger target distance, which likely polluted training data.

## Code Changes
- `halfweg/planner.py`
  - In `search_policy`, changed intermediate first-leg planning from random `b_j in {0,1}` to fixed forward planning (`b=0`).
- `halfweg/train.py`
  - In `collect_self_play`, changed planner calls to always use `b=0`.
  - In `sample_planning_problems`, changed sampled planning bit from random `{0,1}` to fixed `0`.

## Commands
Training:
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/train_halfweg.py --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --epochs 2 --device cuda --max-files 4 --max-levels-per-file 64 --planning-samples 64 --self-play-levels 64 --self-play-steps 32 --ndss 8 --R 3 --save checkpoints/halfweg_attempt3_forward_only.pt
```

Observed training output:
- `epoch=1 examples=64 loss=2.0817 quick_solved=0.000`
- `epoch=2 examples=64 loss=2.8639 quick_solved=0.000`

Evaluation:
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt3_forward_only.pt --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --device cuda --max-files 1 --max-levels-per-file 16 --eval-levels 16 --player-mode random --searches 1

docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt3_forward_only.pt --dataset-root boxoban-levels --split-glob "unfiltered/valid/*.txt" --device cuda --max-files 1 --max-levels-per-file 16 --eval-levels 16 --player-mode random --searches 1
```

## Results
- train: `solve_rate=0.0000` (0/16)
- valid: `solve_rate=0.0000` (0/16)

## Conclusion
Removing unsupported backward-style branches did not improve solve rate in this short run, although it removed a likely source of invalid supervision.
