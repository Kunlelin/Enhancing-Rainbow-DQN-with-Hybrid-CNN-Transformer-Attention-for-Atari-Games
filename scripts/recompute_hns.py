import argparse
import ast
import csv
import json
import os
from typing import Dict, Iterable, Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_WRAPPER_PATH = os.path.join(PROJECT_ROOT, "utils", "env_wrapper.py")


def load_score_tables() -> Tuple[Dict[str, float], Dict[str, float]]:
    with open(ENV_WRAPPER_PATH, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=ENV_WRAPPER_PATH)

    human_scores = None
    random_scores = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "HUMAN_SCORES":
                    human_scores = ast.literal_eval(node.value)
                if isinstance(target, ast.Name) and target.id == "RANDOM_SCORES":
                    random_scores = ast.literal_eval(node.value)

    if human_scores is None or random_scores is None:
        raise RuntimeError("Failed to load HUMAN_SCORES / RANDOM_SCORES from utils/env_wrapper.py")

    return human_scores, random_scores


def normalize_game_key(game: str) -> str:
    return game.lower().strip()


def compute_hns(game: str, raw_score: float,
                human_scores: Dict[str, float],
                random_scores: Dict[str, float]) -> float:
    game_key = normalize_game_key(game)
    h = human_scores.get(game_key, 1.0)
    r = random_scores.get(game_key, 0.0)
    if h == r:
        return 0.0
    return (raw_score - r) / (h - r) * 100.0


def iter_run_dirs(results_dir: str, run_dirs: Iterable[str] | None) -> Iterable[str]:
    if run_dirs:
        for run_dir in run_dirs:
            yield os.path.abspath(run_dir)
        return

    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name)
        if os.path.isdir(path):
            yield path


def recompute_run(run_dir: str,
                  human_scores: Dict[str, float],
                  random_scores: Dict[str, float],
                  dry_run: bool) -> Tuple[int, int]:
    config_path = os.path.join(run_dir, "config.json")
    metrics_path = os.path.join(run_dir, "metrics.csv")
    metrics_json_path = os.path.join(run_dir, "metrics.json")

    if not os.path.exists(config_path) or not os.path.exists(metrics_path):
        return 0, 0

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    game = config["env"]["game"]

    with open(metrics_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows or not fieldnames:
        return 0, 0

    changed = 0
    step_to_hns = {}
    for row in rows:
        raw_score = float(row["raw_score"])
        new_hns = compute_hns(game, raw_score, human_scores, random_scores)
        old_hns = float(row["human_normalized_score"])
        step_to_hns[int(row["step"])] = new_hns
        if abs(old_hns - new_hns) > 1e-9:
            row["human_normalized_score"] = f"{new_hns}"
            changed += 1

    json_changed = 0
    metrics_json = None
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            metrics_json = json.load(f)
        if "human_normalized_score" in metrics_json:
            for item in metrics_json["human_normalized_score"]:
                step, old_value = item
                new_value = step_to_hns.get(int(step), old_value)
                if abs(float(old_value) - float(new_value)) > 1e-9:
                    json_changed += 1

    total_changed = changed or json_changed

    if total_changed and not dry_run:
        with open(metrics_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        if metrics_json is not None and "human_normalized_score" in metrics_json:
            updated_series = []
            for item in metrics_json["human_normalized_score"]:
                step, old_value = item
                updated_series.append([step, step_to_hns.get(int(step), old_value)])
            metrics_json["human_normalized_score"] = updated_series
            with open(metrics_json_path, "w", encoding="utf-8") as f:
                json.dump(metrics_json, f)

    return max(changed, json_changed), len(rows)


def main():
    parser = argparse.ArgumentParser(description="Recompute human_normalized_score in metrics.csv")
    parser.add_argument("--results-dir", type=str, default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--run-dirs", nargs="*", default=None,
                        help="Optional list of specific run directories to fix")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    human_scores, random_scores = load_score_tables()
    results_dir = os.path.abspath(args.results_dir)

    total_changed_runs = 0
    total_changed_rows = 0
    for run_dir in iter_run_dirs(results_dir, args.run_dirs):
        changed, total = recompute_run(run_dir, human_scores, random_scores, args.dry_run)
        if changed:
            total_changed_runs += 1
            total_changed_rows += changed
            action = "would update" if args.dry_run else "updated"
            print(f"{action}: {run_dir} ({changed}/{total} rows)")

    if total_changed_runs == 0:
        print("No HNS changes needed.")
    else:
        print(f"Done: {total_changed_runs} run(s), {total_changed_rows} row(s)")


if __name__ == "__main__":
    main()
