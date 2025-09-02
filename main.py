import cProfile
import pstats
import gymnasium as gym
import snake_ml  # this runs register.py automatically
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch


def train():
    env = make_vec_env(env_str, n_envs=n_envs)
    model = PPO(
        "MlpPolicy",
        env,
        device=device,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        policy_kwargs=policy_kwargs,
        verbose=1,
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
    env_str = "snake_one-hot"
    save_path = f"models/{env_str}/v1.0.zip"
    device = "cuda"
    
    timesteps = 10_000

    n_envs = 48
    n_steps = 1024          # rollout per env
    total_rollout = n_envs * n_steps  # = 16,384
    batch_size = 2048      # divides 16,384 evenly
    n_epochs = 10          # you can try 5–8 if speed is critical

    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512, 256, 128, 64],
            vf=[512, 512, 256, 128, 64]
        ),
        activation_fn=torch.nn.ReLU
    )

    print("=== Starting Snake PPO Script ===")
    train()
    #load_train(env_str)
    test(max_steps=200000, render=True)
    print("=== Script finished ===")
