import os
import time
import argparse
import numpy as np
import torch
import thinker
import gym_sokoban
from thinker.main import Env
from thinker.util import create_flags, init_env_out, create_env_out
from thinker.actor_net import ActorNet

def evaluate_difficulty(difficulty_name, num_episodes=5):
    print(f"\n{'='*50}")
    print(f"Evaluating Difficulty: {difficulty_name}")
    print(f"{'='*50}")
    
    flags = thinker.util.create_setting(args=[])
    flags.name = difficulty_name
    flags.mcts = True
    flags.wrapper_type = 2
    flags.tree_carry = False
    flags.rec_t = 100
    flags.max_depth = -1
    flags.drc = False
    flags.sep_actor_critic = False
    flags.disable_thinker = False
    
    env = Env(
        name=flags.name,
        env_n=1,
        base_seed=42,
        gpu=False,
        train_model=False,
        parallel=False,
        savedir="./logs",
        xpid="eval",
        ckp=False,
        return_x=False,
    )
    
    actor_param = {
        "obs_space": env.observation_space,
        "action_space": env.action_space,
        "flags": flags,
        "tree_rep_meaning": env.get_tree_rep_meaning(),
    }
    
    actor_net = ActorNet(**actor_param)
    actor_state = actor_net.initial_state(batch_size=1)
    
    state, info = env.reset()
    env_out = init_env_out(state, info, flags, actor_net.dim_actions, actor_net.tuple_action)
    
    times = []
    returns = []
    
    for ep in range(num_episodes):
        ep_step = 0
        ep_start_time = time.time()
        
        while True:
            actor_out, actor_state = actor_net(env_out, actor_state, greedy=True)
            action = actor_out.action
            
            state, reward, done, truncated_done, info = env.step(action[0], action[1])
            env_out = create_env_out(action, state, reward, done, truncated_done, info, flags)
            
            if torch.any(env_out.real_done):
                ep_time = time.time() - ep_start_time
                ep_return = env_out.episode_return[env_out.real_done][0, 0].item()
                times.append(ep_time)
                returns.append(ep_return)
                print(f"Episode {ep+1}/{num_episodes}: Time = {ep_time:.2f}s, Return = {ep_return:.2f}")
                break
                
    env.close()
    
    avg_time = np.mean(times)
    avg_return = np.mean(returns)
    print(f"\nResults for {difficulty_name}:")
    print(f"Average Time per Episode: {avg_time:.2f} seconds")
    print(f"Average Return: {avg_return:.2f}")
    
    return avg_time, avg_return

if __name__ == "__main__":
    difficulties = ["Sokoban-v0", "Sokoban-medium-v0", "Sokoban-hard-v0"]
    results = {}
    
    import traceback
    for diff in difficulties:
        try:
            results[diff] = evaluate_difficulty(diff, num_episodes=3)
        except Exception as e:
            print(f"Failed to evaluate {diff}: {e}")
            traceback.print_exc()
            
    print("\n\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    for diff, (avg_time, avg_ret) in results.items():
        print(f"{diff:20s} | Avg Time: {avg_time:6.2f}s | Avg Return: {avg_ret:6.2f}")
