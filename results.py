from collections import defaultdict
import json
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors


def data_per_episode(episode):
    """
    Extracts and structures relevant data from a single episode dictionary.
    """
    food_info = episode['food_info']
    paths = [info['path'] for info in food_info]

    death_info_raw = episode['death_info']
    deprecated = not bool(death_info_raw)

    death_info = {}
    if not deprecated:
        death_info = {
            'score': death_info_raw['score'],
            'death': death_info_raw['death']
        }

    return {
        "paths": paths,
        'deprecated': deprecated,
        'death_info': death_info,
        'total_reward': episode['total_reward']
    }


def aggregate_checkpoints_data(checkpoints):
    """
    Processes a list of checkpoints and yields aggregated metrics for each one.
    """
    for checkpoint in checkpoints:
        # Process each episode within the current checkpoint
        all_episode_data = [data_per_episode(episode) for episode in checkpoint['results']]

        # Separate episodes into valid and deprecated runs
        valid_episodes = [data for data in all_episode_data if not data['deprecated']]
        num_deprecated = len(all_episode_data) - len(valid_episodes)
        
        # Aggregate scores and rewards from valid episodes
        scores = [data['death_info']['score'] for data in valid_episodes]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        
        rewards = [data['total_reward'] for data in valid_episodes]
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

        # Tally death causes and collect death positions
        death_causes = defaultdict(int)
        death_positions = []
        for data in valid_episodes:
            cause = data['death_info']['death']['cause']
            death_causes[cause] += 1

            position = data['death_info']['death']['position']
            if position is not None:
                death_positions.append(position)
        death_causes['deprecated'] = num_deprecated

        NUM_EPISODES = 100
        if not sum(death_causes.values()) == NUM_EPISODES:
            raise RuntimeError(f"More than or less then {NUM_EPISODES} death causes: {sum(death_causes.values())}")

        # Calculate path efficiency ratios for each fruit
        path_ratios_by_fruit = defaultdict(list)
        for episode in valid_episodes:
            for path in episode['paths']:
                optimal = path['optimal_path']
                actual = path['actual_path']
                if optimal > 0:
                    path_ratio = actual / optimal
                    fruit_num = path['fruit_num']
                    path_ratios_by_fruit[fruit_num].append(path_ratio)

        avg_path_ratios = {
            num: sum(ratios) / len(ratios)
            for num, ratios in path_ratios_by_fruit.items()
        }

        # Yield the aggregated data for the current checkpoint
        yield {
            "checkpoint_path": checkpoint['path'],
            "total_episodes": len(all_episode_data),
            "num_deprecated": num_deprecated,
            "score_distribution": scores,
            "mean_score": mean_score,
            "mean_reward": mean_reward,
            "death_causes": dict(death_causes),
            "death_positions": death_positions,
            "avg_path_ratios": avg_path_ratios
        }


def get_steps(data):
    fname = Path(data["checkpoint_path"]).name
    if fname == "model.zip":
        return float("inf")  # put final model last
    if "steps" in fname:
        return int(fname.split("_")[1])  # extract step number
    print('fuck: get_steps')
    return -1  # fallback if format unexpected


def data_to_frames(version: dict) -> pd.DataFrame:
    data = sorted(version, key=get_steps)
    
    df = pd.DataFrame(data)

    # Extract features from path
    path_details = df['checkpoint_path'].str.extract(
        r'models/(?P<model_name>.*?)/(?P<version>.*?)/model_(?P<steps>\d+)_steps.zip'
    )
    path_details['steps'] = pd.to_numeric(path_details['steps'])

    # --- Create Checkpoints DataFrame ---
    death_causes_df = pd.json_normalize(df['death_causes']).add_prefix('death_cause_')
    df_checkpoints = pd.concat([
        df.drop(['checkpoint_path', 'death_causes', 'avg_path_ratios', 'death_positions', 'score_distribution'], axis=1),
        path_details,
        death_causes_df
    ], axis=1)

    # --- Create Death Positions DataFrame ---
    df_positions = df[['checkpoint_path', 'death_positions']].copy()
    df_positions = pd.concat([df_positions, path_details], axis=1)
    df_positions_exploded = df_positions.explode('death_positions')
    df_positions_exploded[['death_x', 'death_y']] = pd.DataFrame(
        df_positions_exploded['death_positions'].tolist(), index=df_positions_exploded.index
    )
    df_positions_tidy = df_positions_exploded.drop(['checkpoint_path', 'death_positions'], axis=1)

    # --- Create Path Ratios DataFrame ---
    df_ratios = df[['checkpoint_path', 'avg_path_ratios']].copy()
    df_ratios = pd.concat([df_ratios, path_details], axis=1)
    df_ratios['path_ratio_items'] = df_ratios['avg_path_ratios'].apply(lambda d: list(d.items()))
    df_ratios_exploded = df_ratios.explode('path_ratio_items')
    df_ratios_exploded[['score', 'avg_path_length']] = pd.DataFrame(
        df_ratios_exploded['path_ratio_items'].tolist(), index=df_ratios_exploded.index
    )
    df_ratios_exploded['score'] = pd.to_numeric(df_ratios_exploded['score'])
    df_ratios_exploded['avg_path_length'] = pd.to_numeric(df_ratios_exploded['avg_path_length'])
    df_path_ratios_tidy = df_ratios_exploded.drop(['checkpoint_path', 'avg_path_ratios', 'path_ratio_items'], axis=1)

    # --- Create Score Distribution DataFrame ---
    df_scores = df[['checkpoint_path', 'score_distribution']].copy()
    df_scores = pd.concat([df_scores, path_details], axis=1)
    df_scores_exploded = df_scores.explode('score_distribution')
    df_scores_exploded.rename(columns={'score_distribution': 'score'}, inplace=True)
    df_scores_exploded['score'] = pd.to_numeric(df_scores_exploded['score'])
    df_scores_tidy = df_scores_exploded.drop(['checkpoint_path'], axis=1)


    return df_checkpoints, df_positions_tidy, df_path_ratios_tidy, df_scores_tidy


def _plot_heatmaps(path_ratios_df: pd.DataFrame):
    """Generates and saves heatmaps for average path length per score."""
    print("\n--- Generating Heatmaps ---")
    for (model_name, version), group_df in path_ratios_df.groupby(['model_name', 'version']):
        print(f"Processing heatmap for model: {model_name} (v{version})...")

        group_df_copy = group_df.copy()
        unique_steps = sorted(group_df_copy['steps'].unique())
        step_to_index_map = {step: i + 1 for i, step in enumerate(unique_steps)}
        group_df_copy['checkpoint_index'] = group_df_copy['steps'].map(step_to_index_map)
        
        pivot_df = group_df_copy.pivot(index='score', columns='checkpoint_index', values='avg_path_length')

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_df,
            cmap='viridis',
            annot=False,
            norm=mcolors.PowerNorm(gamma=0.5)
        )
        plt.gca().invert_yaxis()
        plt.title(f'Avg Path Length | Model: {model_name} {version}', fontsize=16)
        plt.xlabel('Checkpoint Number')
        plt.ylabel('Score')
        
        output_filename = f'graphs/heatmap_{model_name}_{version}.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graph saved to {output_filename}")


def _plot_avg_score(checkpoints_df: pd.DataFrame):
    """Generates and saves a line plot for the average score over training steps."""
    print("\n--- Generating Average Score Plot ---")
    checkpoints_df['model_version'] = checkpoints_df['model_name'] + ' ' + checkpoints_df['version']
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=checkpoints_df, x='steps', y='mean_score', hue='model_version', marker='o')
    
    plt.title('Average Score vs. Training Steps', fontsize=16)
    plt.xlabel('Training Steps')
    plt.ylabel('Mean Score')
    plt.grid(True)
    plt.legend(title='Model Version')
    
    output_filename = 'graphs/avg_score_over_time.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graph saved to {output_filename}")


def _plot_score_stability(df: pd.DataFrame):
    """Generates and saves a line plot showing score trend with a confidence interval."""
    print("\n--- Generating Score Stability Plot ---")
    df['model_version'] = df['model_name'] + ' ' + df['version']

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='steps', y='score', hue='model_version', errorbar=('ci', 95))
    
    plt.title('Score Trend with 95% Confidence Interval', fontsize=16)
    plt.xlabel('Training Steps')
    plt.ylabel('Score')
    plt.grid(True, axis='y')
    plt.legend(title='Model Version')
    
    output_filename = 'graphs/score_stability_over_time.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graph saved to {output_filename}")


def _plot_death_causes(checkpoints_df: pd.DataFrame):
    """Generates and saves stacked area plots for death causes over time."""
    print("\n--- Generating Death Cause Analysis Plots ---")
    death_cause_cols = [col for col in checkpoints_df.columns if col.startswith('death_cause_')]
    
    for (model_name, version), group_df in checkpoints_df.groupby(['model_name', 'version']):
        print(f"Processing death cause plot for model: {model_name} (v{version})...")
        
        plot_df = group_df.set_index('steps')[death_cause_cols].sort_index()
        plot_df.columns = [c.replace('death_cause_', '') for c in plot_df.columns]

        plt.figure(figsize=(12, 8))
        plot_df.plot(kind='area', stacked=True, figsize=(12, 8), ax=plt.gca())

        plt.title(f'Death Causes Over Time | Model: {model_name} {version}', fontsize=16)
        plt.xlabel('Training Steps')
        plt.ylabel('Number of Occurrences')
        plt.legend(title='Death Cause')
        plt.grid(True, axis='y')
        
        output_filename = f'graphs/death_causes_{model_name}_{version}.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graph saved to {output_filename}")


def _plot_avg_reward(checkpoints_df: pd.DataFrame):
    """Generates and saves a line plot for the average reward over training steps."""
    print("\n--- Generating Average Reward Plot ---")
    checkpoints_df['model_version'] = checkpoints_df['model_name'] + ' ' + checkpoints_df['version']
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=checkpoints_df, x='steps', y='mean_reward', hue='model_version', marker='o')
    
    plt.title('Average Reward vs. Training Steps', fontsize=16)
    plt.xlabel('Training Steps')
    plt.ylabel('Mean Reward')
    plt.grid(True)
    plt.legend(title='Model Version')
    
    output_filename = 'graphs/avg_reward_over_time.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graph saved to {output_filename}")


def _plot_path_ratio_vs_score(path_ratios_df: pd.DataFrame):
    """Generates scatter plots of average path length vs. score."""
    print("\n--- Generating Path Ratio vs. Score Plots ---")
    for (model_name, version), group_df in path_ratios_df.groupby(['model_name', 'version']):
        print(f"Processing path ratio vs. score plot for model: {model_name} (v{version})...")
        
        plt.figure(figsize=(12, 8))
        
        # Create a scatter plot, coloring points by the number of training steps
        sns.scatterplot(
            data=group_df,
            x='score',
            y='avg_path_length',
            hue='steps',
            palette='viridis', # Use a sequential colormap
            s=50, # size of points
            alpha=0.7
        )
        
        plt.title(f'Avg Path Length vs. Score | Model: {model_name} {version}', fontsize=16)
        plt.xlabel('Score')
        plt.ylabel('Average Path Length')
        plt.grid(True)
        plt.legend(title='Training Steps')
        
        output_filename = f'graphs/path_ratio_vs_score_{model_name}_{version}.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graph saved to {output_filename}")


def learning_graphs(data):
    """
    Processes model training data to generate and save a series of analytical graphs.

    This function aggregates data from multiple model checkpoints, transforms it into
    pandas DataFrames, and then generates several visualizations including:
    - Heatmaps of path length vs. score.
    - Line plots of average score over time.
    - Line plots of score stability with confidence intervals.
    - Stacked area plots of death causes.

    Args:
        data: A list of dictionaries, where each dictionary represents a model's
              raw data to be processed.
    """
    # --- Step 1: Data Aggregation and Processing ---
    aggregated_data = []
    for model in data:
        aggregated_data.append(list(aggregate_checkpoints_data(model)))

    # Lists to hold the dataframes from each checkpoint
    checkpoints_list, positions_list, ratios_list, scores_list = [], [], [], []

    # Process each checkpoint dictionary
    for checkpoint_data in aggregated_data:
        df_c, df_p, df_r, df_s = data_to_frames(checkpoint_data)
        checkpoints_list.append(df_c)
        positions_list.append(df_p)
        ratios_list.append(df_r)
        scores_list.append(df_s)

    # Concatenate all dataframes into final, aggregated ones
    all_checkpoints = pd.concat(checkpoints_list, ignore_index=True)
    all_positions = pd.concat(positions_list, ignore_index=True)
    all_path_ratios = pd.concat(ratios_list, ignore_index=True)
    all_scores = pd.concat(scores_list, ignore_index=True)

    print("--- Aggregated Checkpoint DataFrame ---")
    print(all_checkpoints)

    print("\n--- Aggregated Score Distribution DataFrame ---")
    print(all_scores.head())

    # --- Step 2: Generate and Save Graphs ---
    # Create the output directory if it doesn't exist to prevent errors
    import os
    os.makedirs('graphs', exist_ok=True)
    
    _plot_heatmaps(all_path_ratios)
    _plot_avg_score(all_checkpoints)
    _plot_score_stability(all_scores)
    _plot_death_causes(all_checkpoints)
    _plot_avg_reward(all_checkpoints)
    _plot_path_ratio_vs_score(all_path_ratios)
    
    print("\nAll graphs have been generated successfully.")


def result_graphs(data):
    pass


if __name__ == '__main__':
    with open('new-results.json', 'r') as file:
        data = json.load(file)

    #sorted_data = sort_by_steps(data)
    learning_graphs(data)
    result_graphs(data)
