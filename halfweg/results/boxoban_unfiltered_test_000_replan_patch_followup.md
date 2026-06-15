# Replanning Patch Follow-up (Boxoban 000)

## What Was Changed
This patch applied the requested priority fixes for closed-loop replanning only.

1. Box-centric mismatch metric
- File: `halfweg/src/hw_impl/evaluate.py`
- `_compute_state_mismatch()` now computes mismatch primarily from the box channel (channel index 2), reducing sensitivity to player micro-position drift.

2. Multi-target traversal restored
- File: `halfweg/src/hw_impl/evaluate.py`
- Removed the previous representative single-target simplification.
- Replanning loop now iterates over all target candidates again.

3. Stall-based trigger added
- File: `halfweg/src/hw_impl/evaluate.py`
- Added `replan_stall_steps` trigger: force replanning when boxes do not change for N steps.
- `replan_every_actions` is now optional (`>0` only); set to 0 to disable unconditional periodic replanning.

4. Updated config for trigger policy
- File: `halfweg/configs/evaluate_boxoban_unfiltered_test_000_gpu_replan_quick.yaml`
- `replan_every_actions: 0`
- `replan_mismatch_threshold: 0.2`
- `replan_stall_steps: 10`

## Validation Status
- Static validation: no errors reported in `evaluate.py`.
- Runtime behavior: the restored `targets: all` + closed-loop replanning path is substantially heavier than the prior single-target shortcut.
- Multiple evaluation runs were started and then manually stopped due excessive runtime (exit 137 on stopped runs).

Generated logs/configs during this follow-up:
- `halfweg/results/boxoban_unfiltered_test_000_eval_replan_quick.log`
- `halfweg/results/boxoban_unfiltered_test_000_eval_replan_probe.log`
- `halfweg/results/boxoban_unfiltered_test_000_eval_replan_tiny.log`
- `halfweg/configs/evaluate_boxoban_unfiltered_test_000_gpu_replan_probe.yaml`
- `halfweg/configs/evaluate_boxoban_unfiltered_test_000_gpu_replan_tiny.yaml`

## Interpretation
- The requested logic corrections are implemented.
- A completed, comparable post-patch metric set (`solved_mean`, `mse_mean`) for the restored multi-target replanning path has not been obtained yet because runs did not finish within practical runtime before manual stop.

## Recommended Next Execution
To obtain comparable numbers without changing algorithm logic:
1. Run the updated quick config as a detached container and let it complete uninterrupted.
2. Add periodic progress logging inside `validate_puzzle_solving__impl` (e.g., every 5 games) to verify active progress during long silent phases.
3. If wall-clock is still too high, benchmark with `n_games_to_solve=10` first, then scale to 100 for final comparison.
