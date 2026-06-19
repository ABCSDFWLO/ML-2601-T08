import argparse
import json
import time
import os
import torch
import thinker.util as util

from thinker.net import ActorNet, ModelNet
from thinker.env import Environment

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
    """
    더미 박스 패딩 없이 원본 맵 그대로를 C++ 환경 강제 규격(1000개)에 맞춰 복사합니다.
    """
    grid = [row[:10].ljust(10, ' ') for row in map_data]
    original_boxes = sum(row.count('$') + row.count('*') for row in grid)
    
    test_dir = "/workspace/csokoban/gym_csokoban/envs/boxoban-levels/unfiltered/test"
    os.makedirs(test_dir, exist_ok=True)
    
    for f in os.listdir(test_dir):
        if f.endswith(".txt"):
            os.remove(os.path.join(test_dir, f))
            
    with open(os.path.join(test_dir, "000.txt"), "w", encoding='utf-8') as f:
        for i in range(1000):
            f.write(f"; {i}\n")
            for line in grid:
                f.write(line + "\n")
            f.write("\n")
            
    return original_boxes

def run_inference(flags, maps):
    device = torch.device("cuda" if torch.cuda.is_available() and not getattr(flags, 'disable_cuda', False) else "cpu")
    print(f"[DEBUG] Using device: {device}")
    
    actor_checkpoint = torch.load(os.path.join(flags.load_checkpoint, "ckp_actor.tar"), map_location=device)
    if "flags" in actor_checkpoint:
        for k, v in actor_checkpoint["flags"].items():
            if k not in ["load_checkpoint", "map_path", "env", "test_rec_t"]:
                setattr(flags, k, v)

    model_checkpoint = torch.load(os.path.join(flags.load_checkpoint, "ckp_model.tar"), map_location=device)

    flags.env = "cSokoban-test-v0"
    results = {"data": {}}
    
    for map_id, map_data in maps.items():
        print(f"\n--- [DEBUG] Processing Map {map_id} ---")
        sys_start_time = time.perf_counter()
        
        original_boxes = duplicate_and_force_map(map_data)
        print(f"[DEBUG] Original logical boxes: {original_boxes}. Injected purely.")
        
        env = Environment(flags, model_wrap=True, env_n=1, device=device)
        
        model_net = ModelNet(obs_shape=env.gym_env_out_shape, num_actions=env.num_actions, flags=flags).to(device)
        model_net.set_weights(model_checkpoint["model_net_state_dict"])
        model_net.eval()

        actor_net = ActorNet(obs_shape=env.model_out_shape if not flags.disable_model else None, gym_obs_shape=env.gym_env_out_shape, num_actions=env.num_actions, flags=flags).to(device)
        actor_net.set_weights(actor_checkpoint["actor_net_state_dict"])
        actor_net.eval()
        
        obs = env.initial(model_net=model_net)
        core_state = actor_net.initial_state(batch_size=1, device=device)
        
        step_count = 0
        solution = []
        status = "failed"
        done = False
        
        boxes_on_target = sum(row.count('*') + row.count('+') for row in map_data)
        
        while not done:
            step_count += 1
            inf_start_time = time.perf_counter()
            with torch.no_grad():
                actor_out, core_state = actor_net(obs, core_state, greedy=getattr(flags, 'greedy', False))
            inf_time_ms = (time.perf_counter() - inf_start_time) * 1000
            
            action_tensor = actor_out.action
            action_idx = action_tensor.item()
            action_str = get_action_string(action_idx)
            
            print(f"[DEBUG] Map {map_id} | Step {step_count} | Action: {action_str} | Inf Time: {inf_time_ms:.2f}ms")
            
            solution.append({
                "step": step_count,
                "forward_time": round(inf_time_ms, 4),
                "action": action_str
            })
            
            env_action = torch.zeros(1, 1, 3, dtype=torch.long, device=device)
            env_action[0, 0, 0] = actor_out.action.item()
            if not getattr(flags, 'disable_model', False):
                env_action[0, 0, 1] = actor_out.im_action.item()
                env_action[0, 0, 2] = actor_out.reset_action.item()
            
            obs = env.step(env_action, model_net=model_net)
            
            step_reward = obs.reward.item()
            if step_reward > 0.5:
                boxes_on_target += 1
            elif step_reward < -0.5:
                boxes_on_target -= 1
            
            if boxes_on_target >= original_boxes:
                done = True
                status = "success"
                print(f"[DEBUG] Map {map_id} solved logically at step {step_count}.")
                break
            
            if obs.real_done.item() or obs.truncated_done.item() or step_count > 120:
                done = True
                print(f"[DEBUG] Map {map_id} finished (failed/truncated) at step {step_count}.")
                break
                
        total_sys_time_ms = (time.perf_counter() - sys_start_time) * 1000
        total_inf_time_ms = sum(step["forward_time"] for step in solution)
        
        map_key = f"{os.path.basename(flags.map_path)}_map_{int(map_id):03d}"
        results["data"][map_key] = {
            "status": status,
            "steps": step_count,
            "inference_time_ms": round(total_inf_time_ms, 2),
            "total_system_time_ms": round(total_sys_time_ms, 2),
            "solution": solution
        }

        env.close()

    print("\n[DEBUG] Inference complete.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_checkpoint", required=True, type=str)
    parser.add_argument("--map_path", required=True, type=str)

    flags, unparsed_args = parser.parse_known_args()
    flags_ = util.parse(unparsed_args)
    for key, value in vars(flags_).items():
        if not hasattr(flags, key):
            setattr(flags, key, value)
            
    maps = parse_boxoban_txt(flags.map_path)
    json_result = run_inference(flags, maps)
    
    with open("results.json", "w") as f:
        json.dump(json_result, f, indent=4)
        print("[DEBUG] Results successfully saved to results.json")