import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

import numpy as np
import cv2
from collections import deque


def _normalize_game_key(game):
    return game.lower().strip()


def get_atari_env_id(game):
    game_key = _normalize_game_key(game)
    game_map = {
        "pong": "ALE/Pong-v5",
        "breakout": "ALE/Breakout-v5",
        "space_invaders": "ALE/SpaceInvaders-v5",
        "seaquest": "ALE/Seaquest-v5",
        "qbert": "ALE/Qbert-v5",
        "beam_rider": "ALE/BeamRider-v5",
        "enduro": "ALE/Enduro-v5",
        "freeway": "ALE/Freeway-v5",
        "frostbite": "ALE/Frostbite-v5",
        "hero": "ALE/Hero-v5",
        "montezuma_revenge": "ALE/MontezumaRevenge-v5",
        "ms_pacman": "ALE/MsPacman-v5",
        "asterix": "ALE/Asterix-v5",
        "bank_heist": "ALE/BankHeist-v5",
        "road_runner": "ALE/RoadRunner-v5",
    }
    return game_map.get(game_key, f"ALE/{game}-v5")


class AtariWrapper:
    def __init__(self, game, frame_stack=4, frame_skip=4, image_size=84,
                 max_episode_length=108000, noop_max=30, clip_reward=True,
                 render_mode=None):
        env_id = get_atari_env_id(game)
        self.env = gym.make(env_id, frameskip=1, render_mode=render_mode)
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.image_size = image_size
        self.max_episode_length = max_episode_length
        self.noop_max = noop_max
        self.clip_reward = clip_reward
        self.frames = deque(maxlen=frame_stack)
        self.lives = 0
        self.episode_length = 0
        self.action_space = self.env.action_space

    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def _get_state(self):
        return np.stack(list(self.frames), axis=0)

    def reset(self):
        obs, info = self.env.reset()
        self.lives = info.get("lives", 0)
        self.episode_length = 0
        noop_actions = np.random.randint(1, self.noop_max + 1)
        for _ in range(noop_actions):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset()
        frame = self._preprocess(obs)
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        return self._get_state()

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            self.episode_length += 1
            if terminated or truncated:
                done = True
                break
            if self.episode_length >= self.max_episode_length:
                done = True
                break
        lives = info.get("lives", 0)
        life_lost = lives < self.lives
        self.lives = lives
        frame = self._preprocess(obs)
        self.frames.append(frame)
        if self.clip_reward:
            total_reward = max(-1.0, min(total_reward, 1.0))
        return self._get_state(), total_reward, done, life_lost

    @property
    def action_size(self):
        return self.action_space.n

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()


HUMAN_SCORES = {
    "alien": 7127.7, "amidar": 1719.5, "assault": 742.0, "asterix": 8503.3,
    "asteroids": 47388.7, "atlantis": 29028.1, "bank_heist": 753.1,
    "battle_zone": 37187.5, "beam_rider": 16926.5, "berzerk": 2630.4,
    "bowling": 160.7, "boxing": 12.1, "breakout": 30.5,
    "centipede": 12017.0, "chopper_command": 7387.8, "crazy_climber": 35829.4,
    "defender": 18688.9, "demon_attack": 1971.0, "double_dunk": -16.4,
    "enduro": 860.5, "fishing_derby": -38.7, "freeway": 29.6,
    "frostbite": 4334.7, "gopher": 2412.5, "gravitar": 3351.4,
    "hero": 30826.4, "ice_hockey": 0.9, "jamesbond": 302.8,
    "kangaroo": 3035.0, "krull": 2665.5, "kung_fu_master": 22736.3,
    "montezuma_revenge": 4753.3, "ms_pacman": 6951.6,
    "name_this_game": 8049.0, "phoenix": 7242.6, "pitfall": 6463.7,
    "pong": 14.6, "private_eye": 69571.3, "qbert": 13455.0,
    "riverraid": 17118.0, "road_runner": 7845.0, "robotank": 11.9,
    "seaquest": 42054.7, "skiing": -4336.9, "solaris": 12326.7,
    "space_invaders": 1668.7, "star_gunner": 10250.0, "surround": 6.5,
    "tennis": -8.3, "time_pilot": 5229.2, "tutankham": 167.6,
    "up_n_down": 11693.2, "venture": 1187.5, "video_pinball": 17667.9,
    "wizard_of_wor": 4756.5, "yars_revenge": 54576.9, "zaxxon": 9173.3,
}

RANDOM_SCORES = {
    "alien": 227.8, "amidar": 5.8, "assault": 222.4, "asterix": 210.0,
    "asteroids": 719.1, "atlantis": 12850.0, "bank_heist": 14.2,
    "battle_zone": 2360.0, "beam_rider": 363.9, "berzerk": 123.7,
    "bowling": 23.1, "boxing": 0.1, "breakout": 1.7,
    "centipede": 2090.9, "chopper_command": 811.0, "crazy_climber": 10780.5,
    "defender": 2874.5, "demon_attack": 152.1, "double_dunk": -18.6,
    "enduro": 0.0, "fishing_derby": -91.7, "freeway": 0.0,
    "frostbite": 65.2, "gopher": 257.6, "gravitar": 173.0,
    "hero": 1027.0, "ice_hockey": -11.2, "jamesbond": 29.0,
    "kangaroo": 52.0, "krull": 1598.0, "kung_fu_master": 258.5,
    "montezuma_revenge": 0.0, "ms_pacman": 307.3,
    "name_this_game": 2292.3, "phoenix": 761.4, "pitfall": -229.4,
    "pong": -20.7, "private_eye": 24.9, "qbert": 163.9,
    "riverraid": 1338.5, "road_runner": 11.5, "robotank": 2.2,
    "seaquest": 68.4, "skiing": -17098.1, "solaris": 1236.3,
    "space_invaders": 148.0, "star_gunner": 664.0, "surround": -10.0,
    "tennis": -23.8, "time_pilot": 3568.0, "tutankham": 11.4,
    "up_n_down": 533.4, "venture": 0.0, "video_pinball": 0.0,
    "wizard_of_wor": 563.5, "yars_revenge": 3092.9, "zaxxon": 32.5,
}


def human_normalized_score(game, raw_score):
    game_key = _normalize_game_key(game)
    h = HUMAN_SCORES.get(game_key, 1.0)
    r = RANDOM_SCORES.get(game_key, 0.0)
    if h == r:
        return 0.0
    return (raw_score - r) / (h - r) * 100.0
