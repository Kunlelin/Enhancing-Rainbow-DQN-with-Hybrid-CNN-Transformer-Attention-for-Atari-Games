import torch
import torch.nn.functional as F
import numpy as np
import copy
from models.rainbow_net import RainbowDQN


class RainbowAgent:
    def __init__(self, action_size, config, device="cuda"):
        self.action_size = action_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = config
        agent_cfg = config["agent"]
        self.atoms = agent_cfg["atoms"]
        self.v_min = agent_cfg["v_min"]
        self.v_max = agent_cfg["v_max"]
        self.discount = agent_cfg["discount"]
        self.multi_step = agent_cfg["multi_step"]
        self.batch_size = agent_cfg["batch_size"]
        self.target_update = agent_cfg["target_update"]
        self.support = torch.linspace(self.v_min, self.v_max, self.atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        self.online_net = RainbowDQN(action_size, config).to(self.device)
        self.target_net = RainbowDQN(action_size, config).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=agent_cfg["learning_rate"],
            eps=agent_cfg["adam_eps"],
        )
        self.steps = 0

    # def act(self, state):
    #     state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    #     self.online_net.eval()
    #     action = self.online_net.get_action(state)
    #     self.online_net.train()
    #     return action

    def act(self, state, eval_mode=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if eval_mode:
            self.online_net.eval()
        else:
            self.online_net.train()
            self.online_net.reset_noise()
        action = self.online_net.get_action(state)
        self.online_net.train()
        return action

    def learn(self, batch):
        states, actions, rewards, next_states, dones, weights, tree_indices, _ = batch
        batch_size = states.size(0)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        self.online_net.reset_noise()
        self.target_net.reset_noise()
        log_probs = self.online_net(states, log=True)
        batch_indices = torch.arange(batch_size, device=self.device)
        log_probs_a = log_probs[batch_indices, actions]
        with torch.no_grad():
            probs_next = self.online_net(next_states)
            q_next = (probs_next * self.support.unsqueeze(0).unsqueeze(0)).sum(2)
            best_actions = q_next.argmax(1)
            target_probs = self.target_net(next_states)
            target_probs_a = target_probs[batch_indices, best_actions]
            discount_n = self.discount ** self.multi_step
            Tz = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * discount_n * self.support.unsqueeze(0)
            Tz = Tz.clamp(self.v_min, self.v_max)
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            l = l.clamp(0, self.atoms - 1)
            u = u.clamp(0, self.atoms - 1)
            target_dist = torch.zeros_like(target_probs_a)
            offset = (torch.arange(batch_size, device=self.device) * self.atoms).unsqueeze(1)
            same_atom = (l == u)
            diff_atom = ~same_atom
            target_dist.view(-1).index_add_(
                0, (l[diff_atom] + offset.expand_as(l)[diff_atom]).view(-1),
                (target_probs_a[diff_atom] * (u.float()[diff_atom] - b[diff_atom])).view(-1)
            )
            target_dist.view(-1).index_add_(
                0, (u[diff_atom] + offset.expand_as(u)[diff_atom]).view(-1),
                (target_probs_a[diff_atom] * (b[diff_atom] - l.float()[diff_atom])).view(-1)
            )
            if same_atom.any():
                target_dist.view(-1).index_add_(
                    0, (l[same_atom] + offset.expand_as(l)[same_atom]).view(-1),
                    target_probs_a[same_atom].view(-1)
                )
        loss = -torch.sum(target_dist * log_probs_a, dim=1)
        weighted_loss = (weights * loss).mean()
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.update_target()
        priorities = loss.detach().cpu().numpy()
        return weighted_loss.item(), priorities, tree_indices

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path):
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = checkpoint["steps"]
