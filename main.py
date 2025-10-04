import os
import mlflow
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from stable_baselines3.common.env_util import make_vec_env
import gittools as git
import snake_ml
from test import test_all

# ----------------
# Training wrapper
# ----------------
def train_model(env_name, save_path, timesteps, checkpoint_steps, ppo_kwargs):
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    # Create a new MLflow Experiment
    mlflow.set_experiment(env_name)

    with mlflow.start_run(run_name=f"{env_name}_v{version}"):
        #log hyperparams
        for k, v in ppo_kwargs.items():
            if isinstance(v, (dict, list)):
                mlflow.log_dict(v, f"{k}.json")
            else:
                mlflow.log_param(k, v)

        
        commit = git.current_commit_hash(checkdirty=True, checktree=True)
        mlflow.log_param("git_commit", commit)

        # build vectorized env
        env = make_vec_env(env_name, n_envs=n_envs)

        # init PPO
        model = PPO("MlpPolicy", env, verbose=0, **ppo_kwargs)

        checkpoint_callback = CheckpointCallback(save_freq = max(checkpoint_steps // n_envs, 1), save_path=f'{save_path}/', name_prefix='model')

        # train
        model.learn(total_timesteps=timesteps, callback=checkpoint_callback)

        # save model + log
        model.save(f'{save_path}/model.zip')
        mlflow.log_artifact(f'{save_path}/model.zip')

        env.close()
        os.remove(f'{save_path}/model.zip')


# ----------------
# Example usage with two models
# ----------------
if __name__ == "__main__":
    import torch
    from multiprocessing import Process

    version = 1.2
    save_path1 = f"models/snake_one-hot/v{version}"
    save_path2 = f"models/snake_int/v{version}"
    device = "cuda"
    n_envs = 48

    onehot_policy = dict(
        net_arch=dict(
            pi=[1024, 512, 512, 256, 64],
            vf=[1024, 512, 512, 256, 64],
        ),
        activation_fn=torch.nn.ReLU
    )

    int_policy = dict(
        net_arch=dict(
            pi=[256, 256, 256, 128, 64],
            vf=[256, 256, 256, 128, 64],
        ),
        activation_fn=torch.nn.ReLU
    )

    base_kwargs = dict(
        n_steps=1024,
        batch_size=2048,
        n_epochs=10,
        learning_rate=0.0001,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        device=device,
    )

    timesteps = 200_000_000
    checkpoint_steps = 5_000_000

    # different kwargs for each policy
    onehot_kwargs = dict(base_kwargs, policy_kwargs=onehot_policy)
    int_kwargs = dict(base_kwargs, policy_kwargs=int_policy)

    jobs = []
    jobs.append(Process(target=train_model, args=("snake_one-hot", save_path1, timesteps, checkpoint_steps), kwargs={'ppo_kwargs': onehot_kwargs}))
    jobs.append(Process(target=train_model, args=("snake_int", save_path2, timesteps, checkpoint_steps), kwargs={'ppo_kwargs': int_kwargs}))

    for j in jobs: j.start()
    for j in jobs: j.join()

    max_steps = 14*14*2
    num_episodes = 100

    model_dir = './models'

    test_all(max_steps, num_episodes, model_dir)