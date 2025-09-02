import gymnasium as gym
import snake_ml  # this runs register.py automatically
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch

def load_train():
    env = make_vec_env(env_str, n_envs=n_envs)
    model = PPO.load(
        path=load_path,
        env=env,
        device=device,
    )

    model.learn(total_timesteps=timesteps)
    model.save(save_path)



def test(max_steps=200, render=True):
    model = PPO.load(save_path)

    env = gym.make(env_str, render_mode="human")
    
    obs, info = env.reset()
    for step in range(max_steps):
        # Model predicts an action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if render:
            env.render()
        
        if terminated or truncated or step == max_steps - 1:
            print(f"Episode ended after {step+1} steps, reward={reward}, info={info}")
            break

    env.close()


if __name__ == "__main__":
    env_str = "Snake-one-hot-v1"
    load_version = 1.0
    load_path = f"hot_large_small-reward_{load_version}.zip"
    save_version = 1.1
    save_path = f"hot_large_small-reward_{save_version}.zip"
    device = "cuda"
    
    timesteps = 10_000_000
    ns_env = 48

    print("=== Starting Snake PPO Script ===")
    load_train()
    test(max_steps=10000, render=True)
    print("=== Script finished ===")
