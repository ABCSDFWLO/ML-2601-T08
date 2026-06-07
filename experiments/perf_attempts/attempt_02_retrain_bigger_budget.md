# Attempt 02 - Retrain with Larger Data and Planning Budget

## Goal
Improve model quality via additional training rather than inference-only tuning.

## What Changed
Training from scratch with a larger training budget:
- `--epochs 2`
- `--max-files 4`
- `--max-levels-per-file 64`
- `--planning-samples 64`
- `--self-play-levels 64`
- `--self-play-steps 32`
- `--ndss 8`
- `--R 3`
- Output checkpoint: `checkpoints/halfweg_attempt2.pt`

## Commands
Training:
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/train_halfweg.py --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --epochs 2 --device cuda --max-files 4 --max-levels-per-file 64 --planning-samples 64 --self-play-levels 64 --self-play-steps 32 --ndss 8 --R 3 --save checkpoints/halfweg_attempt2.pt
```

Observed training output:
- `epoch=1 examples=64 loss=2.6201 quick_solved=0.000`
- `epoch=2 examples=64 loss=2.0614 quick_solved=0.000`

Evaluation (same baseline settings for comparability):
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt2.pt --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --device cuda --max-files 1 --max-levels-per-file 16 --eval-levels 16 --player-mode random --searches 1

docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt2.pt --dataset-root boxoban-levels --split-glob "unfiltered/valid/*.txt" --device cuda --max-files 1 --max-levels-per-file 16 --eval-levels 16 --player-mode random --searches 1
```

## Results
- train: `solve_rate=0.0000` (0/16)
- valid: `solve_rate=0.0000` (0/16)

## Conclusion
Loss decreased across epochs, but solve-rate remained at 0 on the sampled unfiltered train/valid slices. This suggests the policy has not yet crossed the threshold needed for actual puzzle completion.

## Next Action
Increase training horizon further and add curriculum-style training/evaluation (e.g., medium split first), then transfer back to unfiltered.
