# Attempt 08 - Inference Alignment Fixes

## Goal
Improve evaluation-time planning quality without retraining by aligning the search heuristic and replay initialization with Sokoban solve criteria.

## Hypothesis
Two inference-time issues were likely suppressing performance:
1. `distance()` penalized player mismatch even when box placements already matched the goal state.
2. Evaluation replay started almost empty, so hierarchical search had poor intermediate-state candidates.

## Code Changes
- `halfweg/sokoban.py`
  - Changed `distance()` to return `0.0` when box sets match exactly.
  - Reduced player-position weight and increased box/goal mismatch weight.
- `scripts/eval_halfweg.py`
  - Seeded evaluation replay with random-walk reachable states for each level, not just the initial state.

## Commands
Re-evaluation after heuristic/replay fixes:
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt6_curriculum.pt --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --device cuda --max-files 1 --max-levels-per-file 16 --eval-levels 16 --player-mode random --searches 1

docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt6_curriculum.pt --dataset-root boxoban-levels --split-glob "unfiltered/valid/*.txt" --device cuda --max-files 1 --max-levels-per-file 16 --eval-levels 16 --player-mode random --searches 1
```

## Results
- unfiltered train: `solve_rate=0.0000` (0/16)
- unfiltered valid: `solve_rate=0.0000` (0/16)

Additional probe:
- `medium/train` with `--player-mode all --searches 2`: `solve_rate=0.0000` (0/8)
- `medium/valid` with the same setting did not finish in a useful time budget and was stopped.

## Conclusion
The inference heuristic and replay initialization are now more consistent with Sokoban solving, but these fixes alone did not yield solved levels from the current checkpoints.
