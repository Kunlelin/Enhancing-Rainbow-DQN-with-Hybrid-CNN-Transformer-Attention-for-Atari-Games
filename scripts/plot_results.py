import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict


COLORS = {
    "cnn": "#7f7f7f",
    "hybrid": "#d62728",
    "aaconv": "#2ca02c",
}

LABELS = {
    "cnn": "Rainbow (CNN)",
    "hybrid": "Rainbow + CNN-Transformer",
    "aaconv": "Rainbow + AAConv",
}


def load_results(results_dir):
    data = defaultdict(lambda: defaultdict(list))
    for exp_dir in sorted(os.listdir(results_dir)):
        exp_path = os.path.join(results_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        metrics_path = os.path.join(exp_path, "metrics.json")
        config_path = os.path.join(exp_path, "config.json")
        if not os.path.exists(metrics_path) or not os.path.exists(config_path):
            continue
        with open(config_path, "r") as f:
            config = json.load(f)
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        game = config["env"]["game"]
        encoder = config["encoder"]["type"]
        data[game][encoder].append(metrics)
    return data


def smooth(values, window=5):
    if len(values) < window:
        return values
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        smoothed.append(np.mean(values[start:end]))
    return smoothed


def plot_single_game(game, encoder_data, output_dir, metric="raw_score"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for encoder, runs in encoder_data.items():
        all_steps = []
        all_values = []
        for run in runs:
            if metric not in run:
                continue
            steps = [x[0] for x in run[metric]]
            values = [x[1] for x in run[metric]]
            all_steps.append(steps)
            all_values.append(values)
        if not all_values:
            continue
        min_len = min(len(v) for v in all_values)
        steps = all_steps[0][:min_len]
        values_arr = np.array([v[:min_len] for v in all_values])
        mean = np.mean(values_arr, axis=0)
        std = np.std(values_arr, axis=0)
        mean_smooth = smooth(mean)
        frames = [s / 1e6 for s in steps]
        color = COLORS.get(encoder, "#000000")
        label = LABELS.get(encoder, encoder)
        ax.plot(frames, mean_smooth, color=color, label=label, linewidth=2)
        if len(values_arr) > 1:
            ax.fill_between(frames,
                           smooth(mean - std),
                           smooth(mean + std),
                           color=color, alpha=0.15)
    ylabel = "Raw Score" if metric == "raw_score" else "Human Normalized Score (%)"
    ax.set_xlabel("Frames (Millions)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{game.replace('_', ' ').title()}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{game}_{metric}.png"), dpi=150)
    plt.close()


def plot_median_hns(data, output_dir, encoders=None):
    if encoders is None:
        encoders = set()
        for game_data in data.values():
            encoders.update(game_data.keys())
        encoders = sorted(encoders)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for encoder in encoders:
        all_game_hns = []
        common_steps = None
        for game, encoder_data in data.items():
            if encoder not in encoder_data:
                continue
            runs = encoder_data[encoder]
            for run in runs:
                if "human_normalized_score" not in run:
                    continue
                steps = [x[0] for x in run["human_normalized_score"]]
                values = [x[1] for x in run["human_normalized_score"]]
                if common_steps is None:
                    common_steps = steps
                all_game_hns.append(values[:len(common_steps)])
        if not all_game_hns or common_steps is None:
            continue
        min_len = min(len(v) for v in all_game_hns)
        hns_arr = np.array([v[:min_len] for v in all_game_hns])
        median_hns = np.median(hns_arr, axis=0)
        frames = [s / 1e6 for s in common_steps[:min_len]]
        median_smooth = smooth(median_hns)
        color = COLORS.get(encoder, "#000000")
        label = LABELS.get(encoder, encoder)
        ax.plot(frames, median_smooth, color=color, label=label, linewidth=2.5)
    ax.set_xlabel("Frames (Millions)", fontsize=12)
    ax.set_ylabel("Median Human-Normalized Score (%)", fontsize=12)
    ax.set_title("Median Human-Normalized Score Across Games", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="Human level")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "median_hns.png"), dpi=150)
    plt.close()


def plot_games_above_threshold(data, output_dir, thresholds=[20, 50, 100],
                                encoders=None):
    if encoders is None:
        encoders = set()
        for game_data in data.values():
            encoders.update(game_data.keys())
        encoders = sorted(encoders)
    fig, axes = plt.subplots(1, len(thresholds), figsize=(5 * len(thresholds), 5))
    if len(thresholds) == 1:
        axes = [axes]
    for ti, threshold in enumerate(thresholds):
        ax = axes[ti]
        for encoder in encoders:
            common_steps = None
            game_above = []
            for game, encoder_data in data.items():
                if encoder not in encoder_data:
                    continue
                runs = encoder_data[encoder]
                for run in runs:
                    if "human_normalized_score" not in run:
                        continue
                    steps = [x[0] for x in run["human_normalized_score"]]
                    values = [x[1] for x in run["human_normalized_score"]]
                    if common_steps is None:
                        common_steps = steps
                    above = [1 if v >= threshold else 0 for v in values[:len(common_steps)]]
                    game_above.append(above)
            if not game_above or common_steps is None:
                continue
            min_len = min(len(v) for v in game_above)
            arr = np.array([v[:min_len] for v in game_above])
            count = np.sum(arr, axis=0)
            frames = [s / 1e6 for s in common_steps[:min_len]]
            color = COLORS.get(encoder, "#000000")
            label = LABELS.get(encoder, encoder)
            ax.plot(frames, count, color=color, label=label, linewidth=2)
        ax.set_xlabel("Frames (M)", fontsize=10)
        ax.set_ylabel("# Games", fontsize=10)
        ax.set_title(f"Games > {threshold}% Human", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "games_above_threshold.png"), dpi=150)
    plt.close()


def generate_summary_table(data, output_path):
    lines = []
    lines.append("| Game | Encoder | Final Raw Score | Final HNS (%) |")
    lines.append("|------|---------|-----------------|---------------|")
    for game in sorted(data.keys()):
        for encoder in sorted(data[game].keys()):
            runs = data[game][encoder]
            final_scores = []
            final_hns = []
            for run in runs:
                if "raw_score" in run and run["raw_score"]:
                    final_scores.append(run["raw_score"][-1][1])
                if "human_normalized_score" in run and run["human_normalized_score"]:
                    final_hns.append(run["human_normalized_score"][-1][1])
            if final_scores:
                score_str = f"{np.mean(final_scores):.1f} ± {np.std(final_scores):.1f}"
            else:
                score_str = "N/A"
            if final_hns:
                hns_str = f"{np.mean(final_hns):.1f} ± {np.std(final_hns):.1f}"
            else:
                hns_str = "N/A"
            label = LABELS.get(encoder, encoder)
            lines.append(f"| {game} | {label} | {score_str} | {hns_str} |")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="plots")
    args = parser.parse_args()
    data = load_results(args.results_dir)
    if not data:
        print("No results found.")
        return
    print(f"Found data for {len(data)} games")
    for game, encoder_data in data.items():
        print(f"  {game}: {list(encoder_data.keys())}")
        plot_single_game(game, encoder_data, args.output_dir, "raw_score")
        plot_single_game(game, encoder_data, args.output_dir, "human_normalized_score")
    plot_median_hns(data, args.output_dir)
    plot_games_above_threshold(data, args.output_dir)
    generate_summary_table(data, os.path.join(args.output_dir, "summary_table.md"))
    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
