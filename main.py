from stable_baselines3 import PPO
import gymnasium as gym

model = PPO.load("ppo_snake")
env = gym.make("Snake-v0", board_size=14, render_mode="human")
obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()