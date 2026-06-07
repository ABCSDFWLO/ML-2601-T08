# Attempt 06 - Curriculum Training (medium -> unfiltered)

## Goal
Improve convergence by pretraining on easier `medium/train` levels and then finetuning on `unfiltered/train`.

## Hypothesis
Direct training on `unfiltered/train` may be too hard for the current architecture and short training budget. A curriculum from easier levels could provide a better initialization.

## Code Changes
- `scripts/train_halfweg.py`
  - Added optional `--load` checkpoint argument.
  - When provided, training now loads `ma` and `ms` weights before `fit()`.

## Commands
CLI validation:
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace ml2601-halfweg:gpu python scripts/train_halfweg.py --help
```

Stage 1: medium pretraining
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/train_halfweg.py --dataset-root boxoban-levels --split-glob "medium/train/*.txt" --epochs 2 --device cuda --max-files 4 --max-levels-per-file 64 --planning-samples 64 --self-play-levels 64 --self-play-steps 32 --ndss 8 --R 3 --save checkpoints/halfweg_attempt6_medium_pretrain.pt
```

Observed training output:
- `epoch=1 examples=64 loss=2.8190 quick_solved=0.000`
- `epoch=2 examples=64 loss=2.0582 quick_solved=0.000`

Stage 2: unfiltered finetuning
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/train_halfweg.py --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --epochs 2 --device cuda --max-files 4 --max-levels-per-file 64 --planning-samples 64 --self-play-levels 64 --self-play-steps 32 --ndss 8 --R 3 --load checkpoints/halfweg_attempt6_medium_pretrain.pt --save checkpoints/halfweg_attempt6_curriculum.pt
```

Observed finetuning output:
- `epoch=1 examples=64 loss=2.0293 quick_solved=0.000`
- `epoch=2 examples=64 loss=1.8547 quick_solved=0.000`

Evaluation:
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt6_curriculum.pt --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --device cuda --max-files 1 --max-levels-per-file 16 --eval-levels 16 --player-mode random --searches 1

docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt6_curriculum.pt --dataset-root boxoban-levels --split-glob "unfiltered/valid/*.txt" --device cuda --max-files 1 --max-levels-per-file 16 --eval-levels 16 --player-mode random --searches 1
```

## Results
- train: `solve_rate=0.0000` (0/16)
- valid: `solve_rate=0.0000` (0/16)

Additional probe:
- `medium/train` with `--player-mode all --searches 2`: `solve_rate=0.0000` (0/8)
- `medium/valid` with the same stronger inference setting ran too long and was stopped without usable output.

## Conclusion
Curriculum training improved loss relative to earlier runs, but it still did not produce solved levels under the current evaluation budget.
