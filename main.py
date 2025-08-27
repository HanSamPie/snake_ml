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
        #learning_rate=3e-4,
        #gamma=0.99,
        #gae_lambda=0.95,
        #clip_range=0.2,
        #ent_coef=0.01,
        #vf_coef=0.5,
        #max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    # profiler = cProfile.Profile()
    # profiler.enable()

    model.learn(total_timesteps=timesteps)
    model.save(model_path)

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.sort_stats("cumtime").print_stats(50)  


def load_train():
    env = make_vec_env(env_str, n_envs=n_envs)

    model = PPO.load(
        path=model_path,
        env=env,
        device=device
    )

    model.learn(total_timesteps=timesteps)
    model.save(model_path)


def test(max_steps=200, render=True):
    # Load the trained model
    model = PPO.load(model_path)

    # Create a single environment
    env = gym.make(env_str, render_mode="human")
    
    obs, info = env.reset()
    for step in range(max_steps):
        # Model predicts an action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if render:
            env.render()
        
        if terminated or truncated:
            print(f"Episode ended after {step+1} steps, reward={reward}")
            break

    env.close()

if __name__ == "__main__":
    env_str = "Snake-one-hot-v0"
    model_path = "ppo_snake.zip"
    device = "cpu"
    timesteps=10_000

    n_envs = 32
    n_steps = 512          # rollout per env
    total_rollout = n_envs * n_steps  # = 16,384
    batch_size = 1024      # divides 16,384 evenly
    n_epochs = 10          # you can try 5–8 if speed is critical
    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 256, 128],
            vf=[512, 256, 128]
        ),
        activation_fn=torch.nn.ReLU
    )

    
    train()
    load_train()
    test()