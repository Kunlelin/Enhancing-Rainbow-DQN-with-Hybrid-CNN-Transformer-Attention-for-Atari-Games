import os
import sys
import cv2
import yaml
import argparse
from copy import deepcopy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.env_wrapper import AtariWrapper
from agents.rainbow_agent import RainbowAgent


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_video(frames, output_path, fps):
    if not frames:
        raise ValueError("No frames were collected for the video.")
    height, width, _ = frames[0].shape
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(
            "Failed to open video writer. Try a different output path or codec support."
        )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def record_episode(agent, env, max_steps):
    state = env.reset()
    frames = []
    episode_reward = 0.0
    steps = 0

    frame = env.render()
    if frame is not None:
        frames.append(frame)

    done = False
    while not done and steps < max_steps:
        action = agent.act(state, eval_mode=True)
        state, reward, done, _ = env.step(action)
        episode_reward += reward
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        steps += 1

    return frames, episode_reward, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="videos")
    parser.add_argument("--game", type=str, default=None)
    parser.add_argument("--encoder", type=str, default=None,
                        choices=["cnn", "hybrid", "aaconv"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=10000)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config = deepcopy(config)
    if args.game:
        config["env"]["game"] = args.game
    if args.encoder:
        config["encoder"]["type"] = args.encoder
    config["training"]["device"] = args.device

    game = config["env"]["game"]
    encoder = config["encoder"]["type"]
    env = AtariWrapper(
        game,
        frame_stack=config["env"]["frame_stack"],
        frame_skip=config["env"]["frame_skip"],
        image_size=config["env"]["image_size"],
        max_episode_length=config["env"]["max_episode_length"],
        noop_max=config["env"]["noop_max"],
        clip_reward=False,
        render_mode="rgb_array",
    )
    agent = RainbowAgent(env.action_size, config, args.device)
    agent.load(args.checkpoint)

    ensure_dir(args.output_dir)
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]

    for episode_idx in range(args.episodes):
        frames, score, steps = record_episode(agent, env, args.max_steps)
        output_path = os.path.join(
            args.output_dir,
            f"{game}_{encoder}_{stem}_episode{episode_idx + 1}.mp4",
        )
        save_video(frames, output_path, args.fps)
        print(
            f"Saved {output_path} | score={score:.1f} | steps={steps} | frames={len(frames)}"
        )

    env.close()


if __name__ == "__main__":
    main()
