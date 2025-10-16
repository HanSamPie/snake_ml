from stable_baselines3 import PPO
import gymnasium as gym
import snake_ml

def test(max_steps=200, render=True):
    model = PPO.load("./models/snake_one-hot/v2.3/model_199998720_steps.zip")

    env = gym.make("snake_one-hot", render_mode="human")
    
    obs, info = env.reset()
    for step in range(max_steps):
        # Model predicts an action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info["score"] > -1:
            env.render()
        
        if terminated or truncated or step == max_steps - 1:
            print(f"Episode ended after {step+1} steps, reward={reward}, info={info}")
            break

    env.close()

test(10000)