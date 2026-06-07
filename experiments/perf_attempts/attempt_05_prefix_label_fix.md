# Attempt 05 - Fix Auxiliary Prefix Labels

## Goal
Correct a label mismatch in the auxiliary action-model training loss.

## Hypothesis
The auxiliary action loss used an input `(u, ud)` where `ud` is the state after the first `d` actions, but it still used the full plan as the target label. That creates inconsistent supervision because the target for `(u, ud)` should only be the prefix that reaches `ud`.

## Code Changes
- `halfweg/train.py`
  - In `train_epoch`, added `prefix_plan = plan[:self.cfg.d]`.
  - Changed `loss_ma_ref` target from `self._action_target(plan)` to `self._action_target(prefix_plan)`.

## Commands
Training:
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/train_halfweg.py --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --epochs 2 --device cuda --max-files 4 --max-levels-per-file 64 --planning-samples 64 --self-play-levels 64 --self-play-steps 32 --ndss 8 --R 3 --save checkpoints/halfweg_attempt5_prefix_fix.pt
```

Observed training output:
- `epoch=1 examples=64 loss=2.3406 quick_solved=0.000`
- `epoch=2 examples=64 loss=1.9149 quick_solved=0.000`

Evaluation:
```powershell
docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt5_prefix_fix.pt --dataset-root boxoban-levels --split-glob "unfiltered/train/*.txt" --device cuda --max-files 1 --max-levels-per-file 16 --eval-levels 16 --player-mode random --searches 1

docker run --rm --gpus all -v "C:/Users/admin/Documents/ML-2601-T08:/workspace" -w /workspace -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 ml2601-halfweg:gpu python scripts/eval_halfweg.py --checkpoint checkpoints/halfweg_attempt5_prefix_fix.pt --dataset-root boxoban-levels --split-glob "unfiltered/valid/*.txt" --device cuda --max-files 1 --max-levels-per-file 16 --eval-levels 16 --player-mode random --searches 1
```

## Results
- train: `solve_rate=0.0000` (0/16)
- valid: `solve_rate=0.0000` (0/16)

## Conclusion
The label fix corrected the auxiliary objective, but by itself it did not produce a measurable solve-rate gain in this short-run benchmark.
