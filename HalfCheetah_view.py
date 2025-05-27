import gym
import time

# 创建环境
env = gym.make("HalfCheetah-v4", render_mode="human")
obs, info = env.reset(seed=42)  # 设置随机种子

for step in range(1000):
    env.render()  # 渲染场景
    action = env.action_space.sample()  # 随机动作
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Episode finished.")
        obs, info = env.reset()

    time.sleep(0.02)  # 控制动画速度

env.close()
