# Attempt 06 - 50 Epoch Benchmark

## Goal
Benchmark the curriculum-based training setup for a full 50 epochs.

## Note on Code State
This benchmark was run on the current cumulative codebase, which already includes later planner/replay/inference fixes from attempts 7 and 8.

## Training Command
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/train_halfweg.py --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --epochs 50 --device cuda --max-files 2 --max-levels-per-file 32 --planning-samples 32 --self-play-levels 32 --self-play-steps 16 --ndss 4 --R 2 --load checkpoints/halfweg_attempt6_medium_pretrain.pt --save checkpoints/halfweg_attempt6_curriculum_50ep.pt
```

## Training Output
- `epoch=1 examples=32 loss=2.3276 quick_solved=0.000`
- `epoch=10 examples=32 loss=2.3616 quick_solved=0.000`
- `epoch=20 examples=32 loss=2.2071 quick_solved=0.000`
- `epoch=30 examples=32 loss=2.3531 quick_solved=0.000`
- `epoch=40 examples=32 loss=2.3321 quick_solved=0.000`
- `epoch=50 examples=32 loss=2.3027 quick_solved=0.000`
- Saved checkpoint: `checkpoints/halfweg_attempt6_curriculum_50ep.pt`

## Evaluation Commands
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt6_curriculum_50ep.pt --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --device cuda --max-files 1 --max-levels-per-file 16 --eval-levels 16 --player-mode random --searches 1

docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt6_curriculum_50ep.pt --dataset-root boxoban-levels --split-glob "unfiltered/valid/*.txt" --device cuda --max-files 1 --max-levels-per-file 16 --eval-levels 16 --player-mode random --searches 1
```

## Results
- unfiltered train: `solve_rate=0.0000` (0/16)
- unfiltered valid: `solve_rate=0.0000` (0/16)

## Conclusion
Training for 50 epochs lowered loss modestly, but it still did not translate into solved levels on the sampled unfiltered train/valid slices.
