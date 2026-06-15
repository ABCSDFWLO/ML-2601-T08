"""
Sokoban Medium 난이도 문제 생성 및 테스트 스크립트
- medium 난이도 환경 생성
- 랜덤 에이전트로 에피소드 실행
- 결과 시각화 및 통계 출력
"""
import gymnasium as gym
import gym_sokoban
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# Windows 콘솔 UTF-8 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def run_medium_test():
    print("=" * 50)
    print("  Sokoban Medium Difficulty Test")
    print("=" * 50)

    # 1. Medium 난이도 환경 생성
    print("\n[1] Creating Medium difficulty environment...")
    env = gym.make("Sokoban-medium-v0")
    print(f"  [OK] Environment created")
    print(f"  - Action Space: {env.action_space}")
    print(f"  - Observation Space: {env.observation_space}")
    print(f"  - Medium level count: 50,000")

    # 2. 환경 리셋 (문제 생성)
    print("\n[2] Generating problem (reset)...")
    obs, info = env.reset(seed=42)
    print(f"  [OK] Problem generated")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Info: {info}")

    initial_obs = obs.copy()

    # 3. 랜덤 에이전트로 에피소드 실행
    print("\n[3] Running episode with random agent...")
    action_names = ['Noop', 'Up', 'Down', 'Left', 'Right']

    total_reward = 0
    steps = 0
    done = False
    terminated = False
    truncated = False
    rewards_history = []
    frames = [initial_obs]

    while not done and steps < 120:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        rewards_history.append(reward)

        if steps <= 10 or reward != -0.01:
            print(f"  Step {steps:3d}: Action={action_names[action]:5s}, "
                  f"Reward={reward:+.2f}, Total={total_reward:+.2f}, "
                  f"Done={done}")

        if steps in [1, 10, 30, 60, 90, 119] or done:
            frames.append(obs.copy())

    print(f"\n  {'-' * 40}")
    print(f"  Episode Result:")
    print(f"  - Total steps: {steps}")
    print(f"  - Total reward: {total_reward:+.2f}")
    if terminated and not truncated:
        result_str = "SOLVED!"
    elif truncated:
        result_str = "Time limit reached"
    else:
        result_str = "Incomplete"
    print(f"  - Result: {result_str}")
    print(f"  - Reward range: [{min(rewards_history):+.2f}, {max(rewards_history):+.2f}]")

    # 4. 시각화 저장
    print("\n[4] Saving visualizations...")

    # 4a. 초기 상태 이미지
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(initial_obs)
    ax.set_title('Sokoban Medium - Initial State', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'medium_initial.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Initial state saved: {save_path}")

    # 4b. 에피소드 진행 프레임
    n_frames = min(len(frames), 6)
    fig, axes = plt.subplots(1, n_frames, figsize=(4 * n_frames, 4))
    if n_frames == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i < len(frames):
            ax.imshow(frames[i])
            if i == 0:
                ax.set_title('Start', fontsize=11)
            elif i == n_frames - 1:
                ax.set_title(f'End (Step {steps})', fontsize=11)
            else:
                ax.set_title(f'Frame {i}', fontsize=11)
        ax.axis('off')

    plt.suptitle('Sokoban Medium Difficulty - Episode Playthrough',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'medium_episode.png')
    plt.savefig(save_path2, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Episode frames saved: {save_path2}")

    # 4c. 보상 히스토리 그래프
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(range(1, len(rewards_history) + 1), np.cumsum(rewards_history),
            color='#2196F3', linewidth=2, label='Cumulative Reward')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Cumulative Reward', fontsize=12)
    ax.set_title('Sokoban Medium - Reward History (Random Agent)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_path3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'medium_rewards.png')
    plt.savefig(save_path3, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Reward history saved: {save_path3}")

    # 5. 여러 문제 생성 테스트
    print("\n[5] Testing multiple Medium problem generation (5 problems)...")
    for i in range(5):
        obs_i, info_i = env.reset()
        print(f"  Problem {i + 1}: shape={obs_i.shape}, info={info_i}")

    env.close()

    print("\n" + "=" * 50)
    print("  [OK] All tests completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    run_medium_test()
