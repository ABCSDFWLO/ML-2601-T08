import os
import time
import json
import warnings
import argparse
import torch

import thinker.util as util
from thinker.net import ActorNet, ModelNet
from thinker.env import Environment

warnings.filterwarnings("ignore")

WORKSPACE = "/workspace"
TASK_FILE = f"{WORKSPACE}/task.txt"
OUTPUT_JSON = f"{WORKSPACE}/THINKER_results.json"
CHECKPOINT_PATH = "/workspace/trained/trained_backup/base"

def get_action_string(action_idx):
    mapping = {0: 'NOOP', 1: 'U', 2: 'D', 3: 'L', 4: 'R'}
    return mapping.get(action_idx, 'NOOP')

def parse_boxoban_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    maps = {}
    current_map_id = None
    current_map_data = []
    
    for line in lines:
        line = line.strip('\r\n')
        if line.startswith(';'):
            if current_map_id is not None:
                maps[current_map_id] = current_map_data
            current_map_id = line[1:].strip()
            current_map_data = []
        elif line:
            current_map_data.append(line)
            
    if current_map_id is not None:
        maps[current_map_id] = current_map_data
        
    return maps

def duplicate_and_force_map(map_data):
    grid = [row[:10].ljust(10, ' ') for row in map_data]
    test_dir = "/workspace/csokoban/gym_csokoban/envs/boxoban-levels/unfiltered/test"
    os.makedirs(test_dir, exist_ok=True)
    
    with open(os.path.join(test_dir, "000.txt"), "w", encoding='utf-8') as f:
        f.write("; 0\n")
        for line in grid:
            f.write(line + "\n")
        f.write("\n")

def init_model():
    print("[Thinker Daemon] 모델 초기화 및 가중치 파일 로드 중...")
    
    import argparse
    import sys
    import thinker.util as util

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_checkpoint", required=False, type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--map_path", required=False, type=str, default="")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--rec_t", type=int, default=20)
    parser.add_argument("--test_rec_t", type=int, default=-1)

    fake_cli_args = ["--load_checkpoint", CHECKPOINT_PATH]
    flags, unparsed_args = parser.parse_known_args(args=fake_cli_args)
    
    flags_ = util.parse(unparsed_args)
    if flags_:
        for key, value in vars(flags_).items():
            if not hasattr(flags, key):
                setattr(flags, key, value)
                
    # inference.py와 동일하게 테스트 환경 규격 강제 고정
    flags.env = "cSokoban-test-v0"
    
    if not hasattr(flags, 'disable_model'): setattr(flags, 'disable_model', False)
    if not hasattr(flags, 'disable_cuda'): setattr(flags, 'disable_cuda', False)

    device = torch.device("cuda" if torch.cuda.is_available() and not flags.disable_cuda else "cpu")
    print(f"[Thinker Daemon] 사용 장치: {device}")

    # 가중치 데이터를 디스크에서 RAM으로 1회만 로드하여 I/O 병목 제거
    actor_checkpoint = torch.load(os.path.join(flags.load_checkpoint, "ckp_actor.tar"), map_location=device)
    model_checkpoint = torch.load(os.path.join(flags.load_checkpoint, "ckp_model.tar"), map_location=device)

    if "flags" in actor_checkpoint:
        for k, v in actor_checkpoint["flags"].items():
            if k not in ["load_checkpoint", "map_path", "env", "test_rec_t"]:
                setattr(flags, k, v)

    print(f"[Thinker Daemon] 모델 가중치 로드 완료. (Greedy: {getattr(flags, 'greedy', False)}, Rec_T: {getattr(flags, 'rec_t', 20)})")
    
    return actor_checkpoint, model_checkpoint, flags, device

def main():
    actor_checkpoint, model_checkpoint, flags, device = init_model()
    print("[Thinker Daemon] 대기 상태 진입 완료. (task.txt 감시 중)")

    while True:
        if os.path.exists(TASK_FILE):
            try:
                with open(TASK_FILE, 'r') as f:
                    map_file_path = f.read().strip()

                if not os.path.exists(map_file_path):
                    time.sleep(0.1)
                    continue

                print(f"[Thinker Daemon] 연산 시작: {map_file_path}")
                base_filename = os.path.basename(map_file_path)
                
                maps = parse_boxoban_txt(map_file_path)
                results_data = {
                    "model": "thinker",
                    "data": {}
                }
                
                for map_id, map_data in maps.items():
                    map_sys_start = time.perf_counter()
                    
                    duplicate_and_force_map(map_data)
                    
                    # 각 맵마다 Environment와 네트워크를 새로 인스턴스화하여 내부 상태 트리를 완전 초기화
                    env = Environment(flags, model_wrap=True, env_n=1, device=device)
                    
                    model_net = ModelNet(obs_shape=env.gym_env_out_shape, num_actions=env.num_actions, flags=flags).to(device)
                    model_net.set_weights(model_checkpoint["model_net_state_dict"])
                    model_net.eval()

                    actor_net = ActorNet(obs_shape=env.model_out_shape if not flags.disable_model else None, 
                                         gym_obs_shape=env.gym_env_out_shape, 
                                         num_actions=env.num_actions, 
                                         flags=flags).to(device)
                    actor_net.set_weights(actor_checkpoint["actor_net_state_dict"])
                    actor_net.eval()
                    
                    obs = env.initial(model_net=model_net)
                    core_state = actor_net.initial_state(batch_size=1, device=device)
                    
                    real_step_count = 0
                    accumulated_inference_time = 0.0
                    accumulated_planning_time = 0.0
                    solution = []
                    status = "failed"
                    done = False
                    
                    while not done:
                        actor_start_time = time.perf_counter()
                        with torch.no_grad():
                            actor_out, core_state = actor_net(obs, core_state, greedy=getattr(flags, 'greedy', False))
                        actor_time_ms = (time.perf_counter() - actor_start_time) * 1000
                        
                        env_action = torch.zeros(1, 1, 3, dtype=torch.long, device=device)
                        env_action[0, 0, 0] = actor_out.action.item()
                        if not getattr(flags, 'disable_model', False):
                            env_action[0, 0, 1] = actor_out.im_action.item()
                            env_action[0, 0, 2] = actor_out.reset_action.item()
                        
                        env_start_time = time.perf_counter()
                        obs = env.step(env_action, model_net=model_net)
                        env_time_ms = (time.perf_counter() - env_start_time) * 1000

                        accumulated_inference_time += (actor_time_ms + env_time_ms)
                        accumulated_planning_time += env_time_ms
                        
                        if obs.cur_t.item() == 0:
                            real_step_count += 1
                            action_str = get_action_string(actor_out.action.item())
                            solution.append({
                                "step": real_step_count,
                                "forward_time": round(accumulated_inference_time, 4),
                                "planning_time": round(accumulated_planning_time, 4),
                                "action": action_str
                            })

                            accumulated_inference_time = 0.0
                            accumulated_planning_time = 0.0
                        
                        if obs.real_done.item() or obs.truncated_done.item() or real_step_count >= 120:
                            done = True
                            if obs.reward.item() > 5.0:
                                status = "success"
                            break

                    total_map_sys_time_ms = (time.perf_counter() - map_sys_start) * 1000
                    total_inference_time_ms = sum(step["forward_time"] for step in solution)
                    total_planning_time_ms = sum(step["planning_time"] for step in solution)
                    
                    env.close()

                    # 메모리 누수 및 파편화 방지용 삭제
                    del obs, core_state, env_action, model_net, actor_net, env
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    try:
                        map_key = f"{base_filename}_map_{int(map_id):03d}"
                    except ValueError:
                        map_key = f"{base_filename}_map_{map_id}"

                    results_data["data"][map_key] = {
                        "status": status,
                        "steps": real_step_count,
                        "inference_time_ms": round(total_inference_time_ms, 2),
                        "planning_time_ms": round(total_planning_time_ms, 2),
                        "total_system_time_ms": round(total_map_sys_time_ms, 2),
                        "solution": solution
                    }

                temp_json = OUTPUT_JSON + ".tmp"
                with open(temp_json, 'w', encoding='utf-8') as f:
                    json.dump(results_data, f, ensure_ascii=False, indent=4)
                os.rename(temp_json, OUTPUT_JSON)

                print(f"[Thinker Daemon] 모든 맵 ({len(maps)}개) 순차 연산 완료 및 대기 상태 복귀.")

            except Exception as e:
                print(f"[Thinker Daemon] 처리 중 에러 발생: {e}")
            
            finally:
                if os.path.exists(TASK_FILE):
                    try:
                        os.remove(TASK_FILE)
                    except OSError:
                        pass
                
        time.sleep(0.1)

if __name__ == "__main__":
    main()