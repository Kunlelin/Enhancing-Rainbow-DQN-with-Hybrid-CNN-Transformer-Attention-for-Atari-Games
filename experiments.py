import os
import sys
import yaml
import argparse
import subprocess
from itertools import product


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--games", type=str, nargs="+", default=None)
    parser.add_argument("--encoders", type=str, nargs="+",
                        default=["cnn", "hybrid"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--total-frames", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-dir", type=str, default="results")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.games is None:
        games = config["games"]["benchmark"]
    else:
        games = args.games

    experiments = list(product(games, args.encoders, args.seeds))
    print(f"Total experiments: {len(experiments)}")

    for i, (game, encoder, seed) in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(experiments)}: {game} | {encoder} | seed={seed}")
        print(f"{'='*60}")
        cmd = [
            sys.executable, "train.py",
            "--config", args.config,
            "--game", game,
            "--encoder", encoder,
            "--seed", str(seed),
            "--device", args.device,
            "--log-dir", args.log_dir,
        ]
        if args.total_frames:
            cmd.extend(["--total-frames", str(args.total_frames)])
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
