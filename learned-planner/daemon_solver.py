# 파일명: daemon_solver.py (도커 컨테이너 내부용)
import os
import sys
import types

# 1. JAX 백엔드 환경 변수 강제 할당
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# 2. 라이브러리가 요구하는 캐시 경로 자동 생성 및 링크 (리눅스용)
REAL_CACHE_DIR = "/workspace/.sokoban_cache"
FAKE_CACHE_DIR = "/opt/sokoban_cache"

if not os.path.exists(FAKE_CACHE_DIR):
    os.makedirs(REAL_CACHE_DIR, exist_ok=True)
    try:
        os.symlink(REAL_CACHE_DIR, FAKE_CACHE_DIR)
        print(f"[Daemon] Auto-created symlink: {REAL_CACHE_DIR} -> {FAKE_CACHE_DIR}")
    except OSError as e:
        print(f"[Daemon Warning] Failed to create symlink: {e}")

# 3. 환경 변수와 경로 패치가 완료된 후 외부 라이브러리 로드
import time
import json
import glob
import torch
import warnings
from pathlib import Path

from cleanba.environments import BoxobanConfig
from learned_planner.interp.utils import load_jax_model_to_torch
from huggingface_hub import snapshot_download

warnings.filterwarnings("ignore")

# (parse_boxoban_file, setup_isolated_env 함수는 이전 코드와 동일하게 유지)
def parse_boxoban_file(filepath):
    with open(filepath, 'r') as f: content = f.read()
    maps, current_map, map_id = [], [], None
    for line in content.split('\n'):
        if line.startswith(';'):
            if current_map and map_id is not None: maps.append((map_id, '\n'.join(current_map)))
            map_id = line.replace(';', '').strip()
            current_map = [line]
        elif line.strip(): current_map.append(line)
    if current_map and map_id is not None: maps.append((map_id, '\n'.join(current_map)))
    return maps

def setup_isolated_env(map_content):
    isolated_cache = "/tmp/interactive_sokoban_eval"
    isolated_dir = os.path.join(isolated_cache, "boxoban-levels-master", "unfiltered", "train")
    os.makedirs(isolated_dir, exist_ok=True)
    for f in glob.glob(os.path.join(isolated_dir, "*.txt")): os.remove(f)
    with open(os.path.join(isolated_dir, "000.txt"), 'w') as f: f.write(map_content)
    return isolated_cache

def apply_patch(cell):
    orig_pool = cell.pool_and_project
    def safe_pool(self, to_pool):
        if len(to_pool.shape) == 3: to_pool = to_pool.unsqueeze(0)
        res = orig_pool(to_pool)
        return res.unsqueeze(0) if len(res.shape) == 3 else res
    cell.pool_and_project = types.MethodType(safe_pool, cell)
    
    orig_fwd = cell.forward
    def safe_forward(self, *args, **kwargs):
        fixed_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and len(arg.shape) == 3: fixed_args.append(arg.unsqueeze(0))
            elif isinstance(arg, tuple) and len(arg) == 2 and isinstance(arg[0], torch.Tensor):
                h, c = arg
                fixed_args.append((h.unsqueeze(0) if len(h.shape) == 3 else h, c.unsqueeze(0) if len(c.shape) == 3 else c))
            else: fixed_args.append(arg)
        res = orig_fwd(*fixed_args, **kwargs)
        if isinstance(res, tuple):
            fixed_res = []
            for r in res:
                if isinstance(r, torch.Tensor) and len(r.shape) == 3: fixed_res.append(r.unsqueeze(0))
                elif isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], torch.Tensor):
                    h, c = r
                    fixed_res.append((h.unsqueeze(0) if len(h.shape) == 3 else h, c.unsqueeze(0) if len(c.shape) == 3 else c))
                else: fixed_res.append(r)
            return tuple(fixed_res)
        return res.unsqueeze(0) if isinstance(res, torch.Tensor) and len(res.shape) == 3 else res
    cell.forward = types.MethodType(safe_forward, cell)

def main():
    print("[Daemon] Verifying and downloading required Hugging Face weights...")
    snapshot_download(
        repo_id="AlignmentResearch/learned-planner",
        allow_patterns=[
            "drc33/bkynosqi/cp_2002944000/*", 
            "drc33/bkynosqi/*.json", 
            "drc33/bkynosqi/*.txt"
        ]
    )

    print("[Daemon] Loading model to memory...")
    model_path = Path("/home/dev/.cache/huggingface/hub/models--AlignmentResearch--learned-planner/snapshots/6aa0cb55c7194eae7a032889e10932149224bad7/drc33/bkynosqi/cp_2002944000")
    dummy_cfg = BoxobanConfig(max_episode_steps=120, split='train', num_envs=1, tinyworld_obs=True)
    cfg, policy = load_jax_model_to_torch(model_path, dummy_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)

    for c in policy.features_extractor.cell_list: apply_patch(c)
    print("[Daemon] Ready. Waiting for task.txt...", file=sys.stderr)

    task_file = "/workspace/task.txt"
    output_json = "/workspace/DRC33_results.json"
    
    # 무한 루프로 파일 생성 감시
    while True:
        if not os.path.exists(task_file):
            time.sleep(1)
            continue
            
        with open(task_file, 'r') as f:
            target_path = f.read().strip()
            
        if target_path.lower() == 'exit':
            os.remove(task_file)
            break
            
        if not os.path.exists(target_path):
            os.remove(task_file)
            continue

        target_files = [target_path] if os.path.isfile(target_path) else glob.glob(os.path.join(target_path, "**", "*.txt"), recursive=True)
        
        results_data = {"data": {}}
        if os.path.exists(output_json):
            try:
                with open(output_json, 'r') as f: results_data = json.load(f)
            except: pass

        for file_path in target_files:
            file_name = os.path.basename(file_path)
            for map_id, map_content in parse_boxoban_file(file_path):
                map_key = f"{file_name}_map_{map_id.zfill(3)}"
                if map_key in results_data.get("data", {}): continue

                isolated_cache = setup_isolated_env(map_content)
                env_cfg = BoxobanConfig(max_episode_steps=120, split='train', num_envs=1, tinyworld_obs=True, cache_path=Path(isolated_cache))
                
                try:
                    env = env_cfg.make()
                    obs, info = env.reset()
                except: continue

                num_layers = len(policy.features_extractor.cell_list)
                lstm_states = [(torch.zeros((1, 32, 10, 10), dtype=torch.float32).to(device), torch.zeros((1, 32, 10, 10), dtype=torch.float32).to(device)) for _ in range(num_layers)]
                episode_starts = torch.ones((1,), dtype=torch.bool).to(device)

                step_count, total_time = 0, 0.0
                done, solved = False, False
                step_outputs = []

                with torch.no_grad():
                    while not done and step_count < 120:
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device) / 255.0
                        t0 = time.perf_counter()
                        features, lstm_states = policy.features_extractor(obs_tensor, lstm_states, episode_starts)
                        latent_pi, latent_vf = policy.mlp_extractor(features)
                        policy_logits = policy.action_net(latent_pi)
                        value = policy.value_net(latent_vf)
                        forward_time = time.perf_counter() - t0
                        total_time += forward_time
                        
                        action = torch.argmax(policy_logits, dim=-1).cpu().numpy()
                        action_map = {0: "U", 1: "D", 2: "L", 3: "R"}
                        
                        step_outputs.append({"step": step_count + 1, "forward_time": forward_time, "action": action_map.get(int(action[0]), str(int(action[0]))), "policy_logits": policy_logits[0].tolist(), "value": value[0].item()})

                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_starts = torch.zeros((1,), dtype=torch.bool).to(device)

                        if reward[0] > 0: solved = True
                        done = terminated[0] or truncated[0]
                        step_count += 1
                env.close()

                results_data["data"][map_key] = {"status": "success" if solved else "failed", "solve_time_ms": total_time * 1000, "steps": step_count, "solution": step_outputs}

        # 결과 저장 및 작업 파일 삭제 (완료 신호)
        temp_json = output_json + ".tmp"
        with open(temp_json, 'w') as f: json.dump(results_data, f, indent=4)
        os.replace(temp_json, output_json)
        os.remove(task_file)

if __name__ == "__main__":
    main()