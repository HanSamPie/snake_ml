import gittools as git
import json

def version_control():
    commit = git.current_commit_hash(checkdirty=True, checktree=True)
    
    
    
    
    
    data = [
        {
            "version": 1.0,
            "time_steps": 10000,
            "commit_hash": "123h4jk32h4",
            "env_str":  "snake_one-hot",
            "n_envs": 48,
            "n_steps": 1024,         
            "batch_size": 2048,     
            "n_epochs": 10,    
            "hyper_parameter": {
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "vf_coef": 0.5,    
                "policy_kwargs": {
                    "net_arch": {
                        "pi": [512, 512, 256, 128, 64],
                        "vf": [512, 512, 256, 128, 64]
                    },
                    "activation_fn": "torch.nn.ReLU"
                }
            }
        }
    ]

    json_str = json.dumps(data, indent=4)
    with open("sample.json", "w") as f:
        f.write(json_str)
