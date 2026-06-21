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

5. Progress monitoring + call-throttle
- File: `halfweg/src/hw_impl/evaluate.py`
- Added progress logs inside `_solve__closed_loop_replan`:
	- game-level progress (`[Replan] Planning Game ...`)
	- target-level progress (`[Replan] Game X: target Y/Z`)
- Added `min_commit_steps` so replanning triggers are not checked too early in each plan segment.

## Validation Status
- Static validation: no errors reported in `evaluate.py`.
- Runtime behavior: the restored `targets: all` + closed-loop replanning path is substantially heavier than the prior single-target shortcut.
- Multiple evaluation runs were started and then manually stopped due excessive runtime (exit 137 on stopped runs).

Completed deterministic run (updated logic):
- Config: `halfweg/configs/evaluate_boxoban_unfiltered_test_000_gpu_replan_onegame.yaml`
- Games: `1`
- `n_max_episode_steps`: `100`
- Result:
	- `solved_mean`: `0.0`
	- `mse_mean`: `6.6820987654321`
	- `proposed_plan_length_mean`: `70.88888888888889`
	- `solving time`: `45.78853 sec`

Completed detached benchmark run (updated logic):
- Config: `halfweg/configs/evaluate_boxoban_unfiltered_test_000_gpu_replan_10games.yaml`
- Games: `10`
- `n_max_episode_steps`: `100`
- Result:
	- `solved_mean`: `0.6`
	- `mse_mean`: `4.68904039706398`
	- `proposed_plan_length_mean`: `66.3438202247191`
	- `solved_plan_length_mean`: `34.333333333333336`
	- `solving time`: `458.38278 sec`

Generated logs/configs during this follow-up:
- `halfweg/results/boxoban_unfiltered_test_000_eval_replan_quick.log`
- `halfweg/results/boxoban_unfiltered_test_000_eval_replan_probe.log`
- `halfweg/results/boxoban_unfiltered_test_000_eval_replan_tiny.log`
- `halfweg/results/boxoban_unfiltered_test_000_eval_replan_onegame.log`
- `halfweg/results/boxoban_unfiltered_test_000_eval_replan_10games.log`
- `halfweg/configs/evaluate_boxoban_unfiltered_test_000_gpu_replan_probe.yaml`
- `halfweg/configs/evaluate_boxoban_unfiltered_test_000_gpu_replan_tiny.yaml`
- `halfweg/configs/evaluate_boxoban_unfiltered_test_000_gpu_replan_onegame.yaml`
- `halfweg/configs/evaluate_boxoban_unfiltered_test_000_gpu_replan_10games.yaml`

## Interpretation
- The requested logic corrections are implemented.
- Exact metrics are now confirmed for a completed one-game run with the corrected replanning logic.
- On a completed 10-game detached run, the corrected replanning setup recovered strongly:
	- `solved_mean=0.6` (near baseline `0.61`)
	- `mse_mean=4.6890` (slightly better than baseline `4.7049`)
- The 100-game comparable run remains a long-running task and needs uninterrupted detached execution for final parity comparison.

## Recommended Next Execution
To obtain comparable numbers without changing algorithm logic:
1. Run detached 10-game config first (same logic) and collect stable `solved_mean`/`mse_mean`.
2. Then run detached 100-game config for final baseline-comparable metric.
3. Keep `progress_every_targets` enabled and stream logs via `docker logs -f` to confirm forward progress.
