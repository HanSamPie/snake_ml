import json
from pathlib import Path

with open('./aggregated-data.json', 'r') as file:
    data = json.load(file)


def get_steps(data):
    fname = Path(data["checkpoint_path"]).name
    if fname == "model.zip":
        return float("inf")  # put final model last
    if "steps" in fname:
        return int(fname.split("_")[1])  # extract step number
    print('fuck: get_steps')
    return -1  # fallback if format unexpected

for version in data:
    version = sorted(version, key=get_steps)

    last_episode = version[-1]
    print(last_episode['checkpoint_path'])
    causes = { 'deprecated': 0, 'snake': 0, 'border': 0 }
    scores = []
    for death in last_episode["death_details"]:
        causes[death["cause"]] += 1
        scores.append(death["score"])
    mean_score = sum(scores)/len(scores)
    print(causes)
    print(mean_score)
    # for episode in version:
    #     print(episode['checkpoint_path'])
    #     causes = { 'deprecated': 0, 'snake': 0, 'border': 0 }
    #     scores = []
    #     for death in episode["death_details"]:
    #         causes[death["cause"]] += 1
    #         scores.append(death["score"])
    #     mean_score = sum(scores)/len(scores)