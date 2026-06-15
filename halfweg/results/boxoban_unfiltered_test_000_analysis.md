# Halfweg Evaluation Analysis: boxoban-levels/unfiltered/test/000.txt

## 1) Objective
Run the current Halfweg checkpoint in a GPU-enabled Docker container, feed it `boxoban-levels/unfiltered/test/000.txt`, and analyze the outcome.

## 2) Docker Environment
- Image: `halfweg-gpu-eval:latest`
- Base image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`
- Runtime: Docker with `--gpus all`
- Working directory in container: `/workspace/halfweg`

### Image build definition
- Dockerfile: `halfweg/Dockerfile.gpu`

## 3) Evaluation Config Used
Primary completed run used this config:
- `halfweg/configs/evaluate_boxoban_unfiltered_test_000_gpu_quick.yaml`

Key parameters:
- Level source: `/workspace/boxoban-levels/unfiltered/test/000.txt`
- Device: `gpu`
- Method: `one_shot`
- Policy: `last` (resolved to `Policy5 (layer_idx=6)`)
- Towards/Away mode: `towards`
- Games evaluated: `100`

## 4) Execution Artifacts
- Raw quick-run log: `halfweg/results/boxoban_unfiltered_test_000_eval_quick.log`
- Container id record (earlier long run): `halfweg/results/halfweg_eval_000_container_id.txt`

## 5) Result Summary (Completed Run)
From the completed quick run:

- `method`: `one_shot`
- `towards_or_away`: `True` (towards)
- `policy`: `Policy5 (layer_idx=6)`
- `solved_mean`: `0.61`
- `mse_mean`: `4.704882831167356`
- `games_cnt`: `100`
- `proposed_plan_length_mean`: `98.82785808147175`
- `solved_plan_length_mean`: `102.37704918032787`

Timing checkpoints:
- Checkpoint loaded: `1.82869 sec`
- Envs loaded: `7.13355 sec`
- Solving: `18.73796 sec`

Approximate end-to-end runtime from printed checkpoints: about `27.7 sec`.

## 6) Interpretation
- The model solved about 61% of sampled episodes from `000.txt` under this setup.
- Mean plan length is around 99 steps, while solved trajectories average ~102 steps; this implies the policy often proposes plans close to executed successful trajectory lengths, but not always tightly optimized.
- `mse_mean` around 4.70 indicates moderate state mismatch at episode end on average for this evaluation mode.

## 7) Notes and Limitations
- A full 1000-game run config (`evaluate_boxoban_unfiltered_test_000_gpu.yaml`) was launched earlier but was interrupted during stabilization and container deduplication (exit 137 after manual stop), so it is not treated as a valid final result.
- The reported metrics in this analysis are from the completed 100-game quick run only.

## 8) Repro Command
From repository root:

```powershell
docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace/halfweg halfweg-gpu-eval python src/go_halfweg.py configs/evaluate_boxoban_unfiltered_test_000_gpu_quick.yaml
```

## 9) Concrete Improvement Plan to Raise solved_mean

The following actions are prioritized from highest expected impact to lowest implementation risk, based on the observed run (`solved_mean=0.61`, `mse_mean=4.70`) and the documented HalfWeg behavior.

### A. Add closed-loop replanning (replace pure one_shot execution)

Problem signal:
- `one_shot` fixes the full hierarchical plan before acting.
- Final-state mismatch (`mse_mean=4.70`) indicates plan drift between predicted landmarks and reached states.

Implementation detail:
- Recompute high-level landmarks from the current real state every `K` primitive actions (start with `K in {4, 8, 12}`).
- Trigger immediate replanning if either condition is true:
	- normalized state mismatch exceeds `tau_mse` (start with `tau_mse in {2.5, 3.5, 4.5}`), or
	- no box progress for `T_stall` steps (start with `T_stall in {8, 12}`).
- Keep low-level action horizon short during replanning windows (for example 4 to 8 actions) to limit compounding error.

Success criteria:
- Primary: increase `solved_mean`.
- Secondary: reduce `mse_mean` and reduce variance in solved trajectory length.

### B. Strengthen low-level control (ModelActions / PLTA) for player positioning

Problem signal:
- Documented behavior: box motion can be reasonable, but player placement errors frequently break feasibility.

Implementation detail:
- Continue PLTA-focused fine-tuning with batches biased toward states where:
	- player is blocked from required push-side access,
	- player path-to-push-position is long or fragile,
	- recent rollout ended with deadlock or no-progress.
- Add auxiliary objectives:
	- reachability loss: penalize action sequences that move to states with fewer reachable push positions,
	- deadlock-aware penalty: higher loss weight when actions enter known local deadlock motifs.
- Increase PLTA update share versus PLHW in late training (for example PLTA:PLHW update ratio from 1:1 to 2:1 in the final phase).

Success criteria:
- Higher success in episodes previously failing after near-correct box arrangement.
- Fewer no-progress terminations under the same action budget.

### C. Improve high-level landmark feasibility (ModelLandmark / PLHW)

Problem signal:
- High-level landmarks can be box-centric while ignoring player feasibility.

Implementation detail:
- Extend state encoding with player mobility channels:
	- reachable area mask from current player position,
	- push-frontier mask (cells where a valid push can be initiated),
	- optional shortest-path distance transform to nearest push-capable cell.
- During landmark training, add feasibility regularization:
	- penalize predicted landmarks that require unreachable player-side access for planned box pushes.
- At inference, reject top-level landmark proposals failing a lightweight feasibility check and resample top-N alternatives.

Success criteria:
- Lower mismatch between planned landmarks and executable transitions.
- Better solved rate on levels where failure previously occurred after correct macro intent but invalid micro setup.

### D. Optimize plan length and stop/idle behavior

Problem signal:
- Proposed and solved plan lengths are close but not tightly efficient; conservative bursts and idle/stop behavior can waste steps.

Implementation detail:
- Rebalance reward/loss terms for action efficiency:
	- mild per-step penalty,
	- stronger penalty for premature `stop`,
	- bonus for completing subgoal within expected micro-horizon.
- Add a consistency check before accepting `stop`:
	- accept `stop` only if local subgoal distance is below epsilon or no legal improvement action exists.
- Run targeted sweep of stop threshold and step penalty coefficients.

Success criteria:
- Reduced wasted steps and fewer early-stop failures.
- Improved solved rate at fixed max episode length.

## 10) Recommended Experiment Order (fast to strong)

1. Replanning ablation first (`K`, `tau_mse`, `T_stall`) on the same `000.txt` slice.
2. PLTA-focused fine-tuning with reachability/deadlock auxiliary terms.
3. PLHW input/regularization upgrade with mobility-aware features.
4. Stop/idle reward tuning after A-C are stable.

Suggested decision rule:
- Keep a change only if it improves `solved_mean` by at least +0.03 on repeated runs and does not regress `mse_mean`.
