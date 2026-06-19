import argparse
import os
import torch
import torch.nn.functional as F
import torchvision
import thinker.util as util

from thinker.net import ActorNet, ModelNet
from thinker.env import Environment

def pad_and_force_map(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 0번 맵(첫 번째 맵)만 추출
    map_data = []
    for line in lines:
        line = line.strip('\r\n')
        if line.startswith(';'):
            if map_data: break
            continue
        elif line:
            map_data.append(line)
            
    grid = [list(row[:10].ljust(10, ' ')) for row in map_data]
    original_boxes = sum(row.count('$') + row.count('*') for row in grid)
    
    if original_boxes < 4:
        needed = 4 - original_boxes
        empty_spaces = []
        for r in range(1, 9):
            for c in range(1, 9):
                if grid[r][c] == ' ':
                    empty_spaces.append((r, c))
                    
        for i in range(needed):
            if len(empty_spaces) >= 2:
                br, bc = empty_spaces.pop(0)
                tr, tc = empty_spaces.pop(0)
                grid[br][bc] = '$'
                grid[tr][tc] = '.'
            else:
                grid[1][i+1] = '$'
                grid[2][i+1] = '.'
            
    padded_map = ["".join(row) for row in grid]
    
    test_dir = "/workspace/csokoban/gym_csokoban/envs/boxoban-levels/unfiltered/test"
    os.makedirs(test_dir, exist_ok=True)
    for f in os.listdir(test_dir):
        if f.endswith(".txt"):
            os.remove(os.path.join(test_dir, f))
            
    with open(os.path.join(test_dir, "000.txt"), "w", encoding='utf-8') as f:
        for i in range(1000): # C++ 하드코딩 제약을 맞추기 위해 1000개로 복구
            f.write(f"; {i}\n")
            for line in padded_map:
                f.write(line + "\n")
            f.write("\n")

def run_debug(flags):
    device = torch.device("cuda" if torch.cuda.is_available() and not getattr(flags, 'disable_cuda', False) else "cpu")
    print(f"\n[DEBUG INIT] Device: {device}")
    
    # 모델 가중치 로드 및 flags 병합
    actor_checkpoint = torch.load(os.path.join(flags.load_checkpoint, "ckp_actor.tar"), map_location=device)
    if "flags" in actor_checkpoint:
        for k, v in actor_checkpoint["flags"].items():
            if k not in ["load_checkpoint", "map_path", "env", "test_rec_t"]:
                setattr(flags, k, v)
    model_checkpoint = torch.load(os.path.join(flags.load_checkpoint, "ckp_model.tar"), map_location=device)

    flags.env = "cSokoban-test-v0"
    pad_and_force_map(flags.map_path)
    
    env = Environment(flags, model_wrap=True, env_n=1, device=device)
    
    model_net = ModelNet(obs_shape=env.gym_env_out_shape, num_actions=env.num_actions, flags=flags).to(device)
    model_net.set_weights(model_checkpoint["model_net_state_dict"])
    model_net.eval()

    actor_net = ActorNet(obs_shape=env.model_out_shape if not flags.disable_model else None, gym_obs_shape=env.gym_env_out_shape, num_actions=env.num_actions, flags=flags).to(device)
    actor_net.set_weights(actor_checkpoint["actor_net_state_dict"])
    actor_net.eval()
    
    print("\n[STEP 0] Environment Initialization")
    obs = env.initial(model_net=model_net)
    core_state = actor_net.initial_state(batch_size=1, device=device)
    
    print(f" - obs.gym_env_out shape: {obs.gym_env_out.shape}, dtype: {obs.gym_env_out.dtype}")
    print(f" - obs.gym_env_out min/max: {obs.gym_env_out.min().item()} / {obs.gym_env_out.max().item()}")
    
    # 첫 프레임 시각화 저장 (채널, H, W 변환 처리)
    frame_tensor = obs.gym_env_out[0, 0].float() / 255.0
    torchvision.utils.save_image(frame_tensor, "debug_frame_step0.png")
    print(f" - Saved initial frame to 'debug_frame_step0.png'")
    
    action_mapping = {0: 'NOOP', 1: 'U', 2: 'D', 3: 'L', 4: 'R'}
    
    for step in range(1, 4):
        print(f"\n[STEP {step}] Forward Pass")
        with torch.no_grad():
            actor_out, core_state = actor_net(obs, core_state, greedy=getattr(flags, 'greedy', False))
            
        logits = actor_out.policy_logits[0, 0]
        probs = F.softmax(logits, dim=0)
        action_idx = actor_out.action.item()
        baseline = actor_out.baseline[0, 0].squeeze().tolist() if actor_out.baseline is not None else "None"
        
        print(f" - Raw Policy Logits : {logits.cpu().numpy()}")
        print(f" - Action Probs      : {probs.cpu().numpy()}")
        print(f" - Chosen Action     : {action_mapping[action_idx]} (idx: {action_idx})")
        print(f" - Baseline (Value)  : {baseline}")
        
        env_action = torch.zeros(1, 1, 3, dtype=torch.long, device=device)
        env_action[0, 0, 0] = action_idx
        if getattr(flags, 'disable_model', False) is False:
            env_action[0, 0, 1] = actor_out.im_action.item()
            env_action[0, 0, 2] = actor_out.reset_action.item()
            
        obs = env.step(env_action, model_net=model_net)
        
        print(f" - Reward: {obs.reward.item()}, Real Done: {obs.real_done.item()}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_checkpoint", required=True, type=str)
    parser.add_argument("--map_path", required=True, type=str)
    
    flags, unparsed_args = parser.parse_known_args()
    flags_ = util.parse(unparsed_args)
    for key, value in vars(flags_).items():
        if not hasattr(flags, key):
            setattr(flags, key, value)
            
    run_debug(flags)