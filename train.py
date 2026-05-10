import os
import sys
import yaml
import argparse
import numpy as np
import torch
import random
from copy import deepcopy

from utils.env_wrapper import AtariWrapper, human_normalized_score
from utils.replay_buffer import PrioritizedReplayBuffer
from utils.logger import Logger
from agents.rainbow_agent import RainbowAgent

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(agent, config, game, num_episodes=10):
    env = AtariWrapper(
        game,
        frame_stack=config["env"]["frame_stack"],
        frame_skip=config["env"]["frame_skip"],
        image_size=config["env"]["image_size"],
        max_episode_length=config["env"]["max_episode_length"],
        noop_max=config["env"]["noop_max"],
        clip_reward=False,
    )
    scores = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            # action = agent.act(state)
            action = agent.act(state, eval_mode=True)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        scores.append(total_reward)
    env.close()
    return np.mean(scores), np.std(scores)


def run_logged_evaluation(agent, config, game, eval_episodes, frame, logger, experiment_name):
    mean_score, std_score = evaluate(agent, config, game, eval_episodes)
    hns = human_normalized_score(game, mean_score)
    logger.log(
        step=frame,
        raw_score=mean_score,
        std_score=std_score,
        human_normalized_score=hns,
    )
    print(f"[{experiment_name}] EVAL frame={frame}: "
          f"raw={mean_score:.1f} +/- {std_score:.1f}, HNS={hns:.1f}%")
    return mean_score


def train_single_game(config, game, encoder_type, seed=42, log_dir="results"):
    config = deepcopy(config)
    config["env"]["game"] = game
    config["encoder"]["type"] = encoder_type
    config["training"]["seed"] = seed
    set_seed(seed)
    experiment_name = config.get("logging", {}).get(
        "experiment_name", f"{game}_{encoder_type}_seed{seed}"
    )
    logger = Logger(log_dir, experiment_name)
    logger.save_config(config)
    env = AtariWrapper(
        game,
        frame_stack=config["env"]["frame_stack"],
        frame_skip=config["env"]["frame_skip"],
        image_size=config["env"]["image_size"],
        max_episode_length=config["env"]["max_episode_length"],
        noop_max=config["env"]["noop_max"],
    )
    device = config["training"]["device"]
    agent = RainbowAgent(env.action_size, config, device)
    memory = PrioritizedReplayBuffer(
        capacity=config["agent"]["memory_capacity"],
        priority_exponent=config["agent"]["priority_exponent"],
        priority_weight=config["agent"]["priority_weight"],
        multi_step=config["agent"]["multi_step"],
        discount=config["agent"]["discount"],
        frame_stack=config["env"]["frame_stack"],
    )
    total_frames = config["training"]["total_frames"]
    eval_interval = config["training"]["eval_interval"]
    replay_freq = config["training"]["replay_frequency"]
    learn_start = config["agent"]["learn_start"]
    log_interval = config["training"]["log_interval"]
    eval_episodes = config["training"]["eval_episodes"]
    state = env.reset()
    episode_reward = 0.0
    episode_count = 0
    frame = 0
    best_eval_score = -float("inf")
    last_eval_frame = None
    print(f"[{experiment_name}] Training started, total_frames={total_frames}")
    if eval_interval > total_frames:
        print(f"[{experiment_name}] eval_interval={eval_interval} exceeds "
              f"total_frames={total_frames}; final evaluation will be logged "
              f"at the end of training.")
    while frame < total_frames:
        action = agent.act(state)
        next_state, reward, done, life_lost = env.step(action)
        memory.append(state, action, reward, done or life_lost)
        state = next_state
        episode_reward += reward
        frame += config["env"]["frame_skip"]
        if done:
            state = env.reset()
            episode_count += 1
            episode_reward = 0.0
        if frame >= learn_start and frame % replay_freq == 0:
            memory.anneal_priority_weight(frame, total_frames)
            batch = memory.sample(config["agent"]["batch_size"])
            loss, priorities, tree_indices = agent.learn(batch)
            memory.update_priorities(tree_indices, priorities)
        if frame % log_interval == 0 and frame > 0:
            print(f"[{experiment_name}] frame={frame}/{total_frames}, "
                  f"episodes={episode_count}")
        if frame % eval_interval == 0 and frame > 0:
            mean_score = run_logged_evaluation(
                agent, config, game, eval_episodes, frame, logger, experiment_name
            )
            last_eval_frame = frame
            if mean_score > best_eval_score:
                best_eval_score = mean_score
                agent.save(os.path.join(logger.log_dir, "best_model.pt"))
    if frame > 0 and last_eval_frame != frame:
        mean_score = run_logged_evaluation(
            agent, config, game, eval_episodes, frame, logger, experiment_name
        )
        if mean_score > best_eval_score:
            best_eval_score = mean_score
            agent.save(os.path.join(logger.log_dir, "best_model.pt"))
    agent.save(os.path.join(logger.log_dir, "final_model.pt"))
    logger.save_metrics()
    env.close()
    print(f"[{experiment_name}] Training complete. Best eval: {best_eval_score:.1f}")
    return logger.get_metrics()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--game", type=str, default=None)
    parser.add_argument("--encoder", type=str, default="cnn",
                        choices=["cnn", "hybrid", "aaconv"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-frames", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "results"))
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.game:
        config["env"]["game"] = args.game
    if args.total_frames:
        config["training"]["total_frames"] = args.total_frames
    config["training"]["device"] = args.device
    game = config["env"]["game"]
    train_single_game(config, game, args.encoder, args.seed, args.log_dir)


if __name__ == "__main__":
    main()
