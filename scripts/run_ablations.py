import os
import sys
import yaml
import argparse
from copy import deepcopy
from train import train_single_game


ABLATION_CONFIGS = {
    "hybrid_full": {
        "encoder": {"type": "hybrid"},
        "transformer": {"num_layers": 2, "num_heads": 4, "embed_dim": 64},
    },
    "hybrid_1layer": {
        "encoder": {"type": "hybrid"},
        "transformer": {"num_layers": 1, "num_heads": 4, "embed_dim": 64},
    },
    "hybrid_4layer": {
        "encoder": {"type": "hybrid"},
        "transformer": {"num_layers": 4, "num_heads": 4, "embed_dim": 64},
    },
    "hybrid_dim128": {
        "encoder": {"type": "hybrid"},
        "transformer": {"num_layers": 2, "num_heads": 4, "embed_dim": 128},
    },
    "hybrid_8heads": {
        "encoder": {"type": "hybrid"},
        "transformer": {"num_layers": 2, "num_heads": 8, "embed_dim": 64},
    },
    "hybrid_mean_pool": {
        "encoder": {"type": "hybrid"},
        "transformer": {"num_layers": 2, "num_heads": 4, "embed_dim": 64,
                        "use_cls_token": False},
    },
    "aaconv": {
        "encoder": {"type": "aaconv"},
    },
    "cnn_baseline": {
        "encoder": {"type": "cnn"},
    },
}


def run_ablation(config, game, ablation_name, ablation_overrides, seed, log_dir):
    cfg = deepcopy(config)
    for section, values in ablation_overrides.items():
        if section not in cfg:
            cfg[section] = {}
        cfg[section].update(values)
    exp_name = f"ablation_{game}_{ablation_name}_seed{seed}"
    if "logging" not in cfg:
        cfg["logging"] = {}
    cfg["logging"]["experiment_name"] = exp_name
    print(f"\nRunning ablation: {exp_name}")
    return train_single_game(cfg, game, cfg["encoder"]["type"], seed, log_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--game", type=str, default="breakout")
    parser.add_argument("--ablations", type=str, nargs="+",
                        default=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-frames", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-dir", type=str, default="results")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.total_frames:
        config["training"]["total_frames"] = args.total_frames
    config["training"]["device"] = args.device
    for ablation_name in args.ablations:
        if ablation_name not in ABLATION_CONFIGS:
            print(f"Unknown ablation: {ablation_name}")
            continue
        run_ablation(config, args.game, ablation_name,
                    ABLATION_CONFIGS[ablation_name], args.seed, args.log_dir)


if __name__ == "__main__":
    main()
