import cProfile
import pstats
import gymnasium as gym
import snake_ml  # this runs register.py automatically
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch


def train():
    env = make_vec_env(env_str, n_envs=n_envs)

    print("\n=== Running train() ===")
    print(f"Environment: {env_str}")
    print(f"Device: {device}")
    print(f"Number of environments: {n_envs}")
    print(f"Rollout steps per env (n_steps): {n_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {n_epochs}")
    print(f"Total timesteps: {timesteps}")
    print(f"Policy kwargs: {policy_kwargs}")

    print("Creating PPO model...")
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

    print("Starting model learning...")
    model.learn(total_timesteps=timesteps)
    print(f"Saving model to {model_path}...")
    model.save(model_path)
    print("Training complete!\n")


def load_train():
    env = make_vec_env(env_str, n_envs=n_envs)
    print("\n\n=== Running load_train() ===")
    print(f"Environment: {env_str}")
    print(f"Device: {device}")
    print(f"Number of environments: {n_envs}")
    print(f"Total timesteps: {timesteps}\n\n")
    

    print(f"Loading model from {model_path}...")
    model = PPO.load(
        path=model_path,
        env=env,
        device=device
    )

    print("Continuing model training...")
    model.learn(total_timesteps=timesteps)
    print(f"Saving model to {model_path}...")
    model.save(model_path)
    print("load_train() complete!\n")


def test(max_steps=200, render=True):
    print("\n=== Running test() ===")
    print(f"Environment: {env_str}")
    print(f"Max steps: {max_steps}")
    print(f"Render mode: {'ON' if render else 'OFF'}")

    print(f"Loading trained model from {model_path}...")
    model = PPO.load(model_path)

    print("Creating single test environment...")
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
    print("test() complete!\n")


if __name__ == "__main__":
    env_str = "Snake-one-hot-v3"
    model_path = f"{env_str}.zip"
    device = "cpu"
    timesteps = 10_000

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

    print("=== Starting Snake PPO Script ===")
    #train()
    #load_train()
    test(max_steps=20, render=False)
    print("=== Script finished ===")
