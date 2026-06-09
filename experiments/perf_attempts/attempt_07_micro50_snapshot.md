# Attempt 07 - Micro 50 Epoch Snapshot

## Goal
Reproduce attempt 07 as a 50-epoch micro-benchmark with the same small training/evaluation slice.

## Snapshot State
- Forward-only planning.
- Replay seeded with reachable random-walk states.
- Auxiliary action loss uses the `d`-step prefix target.
- Curriculum loading support via `--load` is enabled in the CLI.
- Reachable trajectory supervision is enabled for training example generation.
- No inference-time replay seeding or distance alignment.

## Training Command
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/train_halfweg.py --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --epochs 50 --device cuda --max-files 1 --max-levels-per-file 8 --planning-samples 8 --self-play-levels 8 --self-play-steps 8 --ndss 2 --R 2 --load checkpoints/halfweg_attempt6_medium_pretrain.pt --save checkpoints/halfweg_attempt07_micro50.pt
```

## Training Output
- `epoch=1 examples=8 loss=2.4363 quick_solved=0.000`
- `epoch=25 examples=8 loss=2.4866 quick_solved=0.000`
- `epoch=50 examples=8 loss=2.5616 quick_solved=0.000`

## Evaluation Command
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt07_micro50.pt --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --device cuda --max-files 1 --max-levels-per-file 8 --eval-levels 8 --player-mode random --searches 1

docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt07_micro50.pt --dataset-root boxoban-levels --split-glob "unfiltered/valid/*.txt" --device cuda --max-files 1 --max-levels-per-file 8 --eval-levels 8 --player-mode random --searches 1
```

## Results
- train: `solve_rate=0.0000` (0/8), `elapsed_sec=8.39`, `avg_sec_per_level=1.05`
- valid: `solve_rate=0.0000` (0/8), `elapsed_sec=8.34`, `avg_sec_per_level=1.04`

## Conclusion
Reachable-trajectory supervision did not solve any sampled levels in the 50-epoch micro-benchmark.
