# Attempt 04 - Replay Seeding with Reachable Random-Walk States

## Goal
Increase replay diversity so planning problems are not dominated by initial states.

## Hypothesis
Replay was seeded only with initial states. That makes sampled subgoals collapse toward a tiny subset of states early in training, reducing useful planning supervision.

## Code Changes
- `halfweg/train.py`
  - In `_seed_replay`, kept the initial state and additionally inserted multiple `random_walk` states per level using increasing step counts up to `self_play_steps`.

## Commands
Training:
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/train_halfweg.py --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --epochs 2 --device cuda --max-files 4 --max-levels-per-file 64 --planning-samples 64 --self-play-levels 64 --self-play-steps 32 --ndss 8 --R 3 --save checkpoints/halfweg_attempt4_replay_seeded.pt
```

Observed training output:
- `epoch=1 examples=64 loss=2.3407 quick_solved=0.000`
- `epoch=2 examples=64 loss=1.9147 quick_solved=0.000`

Evaluation:
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt4_replay_seeded.pt --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --device cuda --max-files 1 --max-levels-per-file 16 --eval-levels 16 --player-mode random --searches 1

docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt4_replay_seeded.pt --dataset-root boxoban-levels --split-glob "unfiltered/valid/*.txt" --device cuda --max-files 1 --max-levels-per-file 16 --eval-levels 16 --player-mode random --searches 1
```

## Results
- train: `solve_rate=0.0000` (0/16)
- valid: `solve_rate=0.0000` (0/16)

## Conclusion
Richer replay seeding reduced loss but still did not translate into solved levels on the sampled train/valid slices.
