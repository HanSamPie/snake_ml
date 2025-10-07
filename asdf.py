import json
from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

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

    df = pd.DataFrame(last_episode["death_details"])

    cause_colors = {
        'snake': 'green',  # A specific shade of red
        'border': 'blue',  # A specific shade of blue
        'deprecated': 'grey'
    }


    plt.figure(figsize=(10, 6))

    # Pass the color dictionary to the 'palette' parameter
    sns.kdeplot(
        data=df,
        x='score',
        hue='cause',
        fill=True,
        palette=cause_colors # This is the key change
    )

    name = last_episode['checkpoint_path'].split("/")
    name = name[2]+ "-" + name[3]
    plt.title(f'Score Distribution by Cause {name}')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.savefig(f'seaborn_kde_plot{name}.png')
    plt.close()

    for death in last_episode["death_details"]:
        causes[death["cause"]] += 1
        scores.append(death["score"])
    mean_score = sum(scores)/len(scores)
    print(causes, mean_score)
    # for episode in version:
    #     print(episode['checkpoint_path'])
    #     causes = { 'deprecated': 0, 'snake': 0, 'border': 0 }
    #     scores = []
    #     for death in episode["death_details"]:
    #         causes[death["cause"]] += 1
    #         scores.append(death["score"])
    #     mean_score = sum(scores)/len(scores)