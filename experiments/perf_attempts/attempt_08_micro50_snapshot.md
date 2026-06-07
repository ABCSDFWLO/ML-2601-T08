# Attempt 08 - Micro 50 Epoch Snapshot

## Goal
Reproduce attempt 08 as a 50-epoch micro-benchmark with the same small training/evaluation slice.

## Snapshot State
- All attempt 07 training-side changes remain in place.
- Inference-time replay seeding is restored in `scripts/eval_halfweg.py`.
- Distance scoring in `halfweg/sokoban.py` is aligned to favor exact box matches.

## Training Command
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/train_halfweg.py --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --epochs 50 --device cuda --max-files 1 --max-levels-per-file 8 --planning-samples 8 --self-play-levels 8 --self-play-steps 8 --ndss 2 --R 2 --load checkpoints/halfweg_attempt6_medium_pretrain.pt --save checkpoints/halfweg_attempt08_micro50.pt
```

## Training Output
- `epoch=1 examples=8 loss=2.4363 quick_solved=0.000`
- `epoch=25 examples=8 loss=2.4936 quick_solved=0.000`
- `epoch=50 examples=8 loss=2.4915 quick_solved=0.000`

## Evaluation Commands
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt08_micro50.pt --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --device cuda --max-files 1 --max-levels-per-file 8 --eval-levels 8 --player-mode random --searches 1

docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt08_micro50.pt --dataset-root boxoban-levels --split-glob "unfiltered/valid/*.txt" --device cuda --max-files 1 --max-levels-per-file 8 --eval-levels 8 --player-mode random --searches 1
```

## Results
- train: `solve_rate=0.0000` (0/8)
- valid: `solve_rate=0.0000` (0/8)

## Conclusion
The inference-alignment changes did not improve solve rate on the sampled micro-benchmark.
