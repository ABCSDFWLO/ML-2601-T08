# Attempt 04 - Micro 200 Epoch Snapshot

## Goal
Run attempt 04 snapshot settings up to 200 epochs.

## Snapshot State
- Forward-only planning.
- Replay seeded with reachable random-walk states.
- No prefix-label fix.
- No reachable-trajectory supervision.
- No inference-time replay seeding or distance alignment.

## Training Command
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/train_halfweg.py --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --epochs 200 --device cuda --max-files 1 --max-levels-per-file 8 --planning-samples 8 --self-play-levels 8 --self-play-steps 8 --ndss 2 --R 2 --save checkpoints/halfweg_attempt04_micro200.pt
```

## Training Output
- `epoch=1 examples=8 loss=0.8659 quick_solved=0.000`
- `epoch=50 examples=8 loss=0.0015 quick_solved=0.000`
- `epoch=100 examples=8 loss=0.0016 quick_solved=0.000`
- `epoch=150 examples=8 loss=0.0011 quick_solved=0.000`
- `epoch=200 examples=8 loss=0.0007 quick_solved=0.000`

## Result
- Checkpoint: `checkpoints/halfweg_attempt04_micro200.pt`
- 200 epochs completed successfully.
- Loss converged to very low values, but `quick_solved` stayed at `0.000` for all epochs.

## Evaluation
- train: `solve_rate=0.0000` (0/8), `elapsed_sec=12.37`, `avg_sec_per_level=1.55`
- valid: `solve_rate=0.0000` (0/8), `elapsed_sec=8.07`, `avg_sec_per_level=1.01`
