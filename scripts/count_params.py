import yaml
import torch
from models.rainbow_net import RainbowDQN


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    action_size = 4
    print(f"{'Encoder':<20} {'Total Params':>15} {'Trainable':>15}")
    print("-" * 55)
    for enc_type in ["cnn", "hybrid", "aaconv"]:
        cfg = dict(config)
        cfg["encoder"] = dict(config["encoder"])
        cfg["encoder"]["type"] = enc_type
        model = RainbowDQN(action_size, cfg)
        total, trainable = count_params(model)
        print(f"{enc_type:<20} {total:>15,} {trainable:>15,}")
    print()
    cfg_hybrid = dict(config)
    cfg_hybrid["encoder"] = dict(config["encoder"])
    cfg_hybrid["encoder"]["type"] = "hybrid"
    for num_layers in [1, 2, 4]:
        cfg_hybrid["transformer"] = dict(config["transformer"])
        cfg_hybrid["transformer"]["num_layers"] = num_layers
        model = RainbowDQN(action_size, cfg_hybrid)
        total, trainable = count_params(model)
        print(f"hybrid (L={num_layers})       {total:>15,} {trainable:>15,}")
    for embed_dim in [32, 64, 128]:
        cfg_hybrid["transformer"] = dict(config["transformer"])
        cfg_hybrid["transformer"]["embed_dim"] = embed_dim
        cfg_hybrid["transformer"]["num_layers"] = 2
        model = RainbowDQN(action_size, cfg_hybrid)
        total, trainable = count_params(model)
        print(f"hybrid (D={embed_dim:<3})      {total:>15,} {trainable:>15,}")


if __name__ == "__main__":
    main()
