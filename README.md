# Enhancing Rainbow DQN with Hybrid CNN-Transformer Attention for Atari Games

This repository contains the code and experiment metrics for a master's dissertation project on replacing the standard Rainbow DQN visual encoder with a lightweight CNN-Transformer hybrid encoder for Atari games.

## Repository Structure

- `train.py`: Main training entry point.
- `experiments.py`: Batch experiment runner.
- `agents/`: Rainbow agent implementation.
- `models/`: CNN, hybrid CNN-Transformer encoder, noisy layers, and Rainbow network definitions.
- `utils/`: Atari environment wrapper, replay buffer, and logging utilities.
- `configs/`: Default training configuration.
- `scripts/`: Plotting, result recomputation, ablation, recording, and utility scripts.
- `results/`: Main 10M-frame experiment metrics and configurations.
- `results_extra_seeds/`: Additional seed experiments.
- `results_long/`: Longer 30M-frame experiment metrics and configurations.
- `results_ablations/`: Qbert ablation experiment metrics and configurations.

Large model checkpoint files (`best_model.pt` and `final_model.pt`) are intentionally omitted from this GitHub upload version because many of them exceed GitHub's normal file-size limit. The included CSV/JSON metrics and configuration files are sufficient for reproducing the reported tables and figures.

## Environment

Install PyTorch separately for the CUDA version of the target machine, then install the remaining Python dependencies:

```bash
pip install -r requirements.txt
```

The Atari environments require `gymnasium` and `ale-py`. A quick environment check can be run with:

```bash
python - <<'PY'
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode=None)
obs, info = env.reset()
print("obs shape:", obs.shape)
print("action n:", env.action_space.n)
env.close()
PY
```

## Training Example

```bash
python train.py \
  --config configs/default.yaml \
  --game qbert \
  --encoder hybrid \
  --seed 42 \
  --device cuda \
  --total-frames 10000000 \
  --log-dir results
```

## Reproducing Figures

```bash
python scripts/plot_thesis_figures.py \
  --results-dir results \
  --extra-results-dir results_extra_seeds \
  --long-results-dir results_long \
  --ablation-results-dir results_ablations \
  --output-dir figures
```

## Notes on Uploaded Data

The result directories keep the experiment metadata and learning curves:

- `config.json` records the run configuration.
- `metrics.csv` stores evaluation scores across training frames.
- `metrics.json` stores the same metrics in JSON form.

Checkpoint files are excluded from this repository version. If model weights need to be shared later, use Git LFS, GitHub Releases, Google Drive, or Zenodo rather than normal Git tracking.
