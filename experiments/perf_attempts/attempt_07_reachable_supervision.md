# Attempt 07 - Reachable Trajectory Supervision

## Goal
Replace weak planner-generated pseudo-labels with exact reachable trajectories from random walks.

## Hypothesis
`build_training_examples()` depended on the current planner to generate plans, so the learner could be trapped by low-quality pseudo-labels. Using reachable trajectories directly should provide cleaner supervision.

## Code Changes
- `halfweg/train.py`
  - Added `_sample_reachable_plan()`.
  - Sampled a replay state as start, rolled out a short sequence of valid actions, and used the resulting trajectory as an exact training example.
  - Changed `build_training_examples()` to use these reachable plans instead of `planner.best_plan_over_hierarchy()` pseudo-labels.

## Commands
Training:
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/train_halfweg.py --dataset-root boxoban-levels --split-glob "medium/train/*.txt" --epochs 2 --device cuda --max-files 4 --max-levels-per-file 64 --planning-samples 64 --self-play-levels 64 --self-play-steps 32 --ndss 8 --R 3 --save checkpoints/halfweg_attempt7_reachable_supervision.pt
```

Observed training output:
- `epoch=1 examples=64 loss=2.4212 quick_solved=0.000`
- `epoch=2 examples=64 loss=2.3695 quick_solved=0.000`

Evaluation:
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt7_reachable_supervision.pt --dataset-root boxoban-levels --split-glob "medium/train/*.txt" --device cuda --max-files 1 --max-levels-per-file 16 --eval-levels 16 --player-mode random --searches 1

docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt7_reachable_supervision.pt --dataset-root boxoban-levels --split-glob "medium/valid/*.txt" --device cuda --max-files 1 --max-levels-per-file 16 --eval-levels 16 --player-mode random --searches 1
```

## Results
- medium train: `solve_rate=0.0000` (0/16)
- medium valid: `solve_rate=0.0000` (0/16)

## Conclusion
Cleaner supervised trajectories did not produce measurable solved levels in this short-run configuration.
