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
