# Attempt 01 - Inference Budget Tuning

## Goal
Improve solve rate without retraining by increasing inference search budget.

## What Changed
- Baseline inference:
  - `--player-mode random`
  - `--searches 1`
- Attempt inference:
  - `--player-mode all`
  - `--searches 4` (train split)
  - For valid split, reduced to `--searches 2` and smaller eval budget to avoid excessive runtime.

No code changes were made in this attempt.

## Commands
Baseline (fast comparable setup):
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_gpu_smoke.pt --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --device cuda --max-files 1 --max-levels-per-file 16 --eval-levels 16 --player-mode random --searches 1

docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_gpu_smoke.pt --dataset-root boxoban-levels --split-glob "unfiltered/valid/*.txt" --device cuda --max-files 1 --max-levels-per-file 16 --eval-levels 16 --player-mode random --searches 1
```

Attempt:
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_gpu_smoke.pt --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --device cuda --max-files 1 --max-levels-per-file 16 --eval-levels 16 --player-mode all --searches 4

docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_gpu_smoke.pt --dataset-root boxoban-levels --split-glob "unfiltered/valid/*.txt" --device cuda --max-files 1 --max-levels-per-file 8 --eval-levels 8 --player-mode all --searches 2
```

## Results
Baseline:
- train: `solve_rate=0.0000` (0/16)
- valid: `solve_rate=0.0000` (0/16)

Attempt:
- train: `solve_rate=0.0000` (0/16)
- valid (reduced budget): `solve_rate=0.0000` (0/8)

## Conclusion
Increasing inference search budget alone did not improve solve rate from the current checkpoint.

## Next Action
Try retraining with larger data/self-play/planning budget and then re-evaluate with the same baseline evaluation settings.
