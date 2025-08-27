import gymnasium as gym
from stable_baselines3 import PPO
import snake_ml  # ensures your env is registered

def test_snake(model_path="ppo_snake.zip", max_steps=200, render=True):
    # Load the trained model
    model = PPO.load(model_path)

    # Create a single environment
    env = gym.make("Snake-one-hot-v0", render_mode="human" if render else None)
    
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
    test_snake()