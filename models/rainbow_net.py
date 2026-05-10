import torch
import torch.nn as nn
import torch.nn.functional as F
from models.noisy_linear import NoisyLinear
from models.encoder import build_encoder, CNNEncoder


class RainbowDQN(nn.Module):
    def __init__(self, action_size, config):
        super().__init__()
        self.action_size = action_size
        self.atoms = config["agent"]["atoms"]
        self.v_min = config["agent"]["v_min"]
        self.v_max = config["agent"]["v_max"]
        self.support = torch.linspace(self.v_min, self.v_max, self.atoms)
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        noisy_std = config["agent"]["noisy_std"]
        hidden = config["encoder"]["hidden_size"]
        self.encoder, enc_output_size = build_encoder(config)
        enc_type = config["encoder"]["type"]
        if enc_type == "cnn":
            self.flatten = True
            feat_size = enc_output_size
        elif enc_type == "hybrid":
            self.flatten = False
            feat_size = enc_output_size
        elif enc_type == "aaconv":
            self.flatten = False
            feat_size = enc_output_size
        else:
            self.flatten = True
            feat_size = enc_output_size
        self.fc_value_hidden = NoisyLinear(feat_size, hidden, noisy_std)
        self.fc_value = NoisyLinear(hidden, self.atoms, noisy_std)
        self.fc_advantage_hidden = NoisyLinear(feat_size, hidden, noisy_std)
        self.fc_advantage = NoisyLinear(hidden, action_size * self.atoms, noisy_std)

    def forward(self, x, log=False):
        feat = self.encoder(x)
        if self.flatten and feat.dim() > 2:
            feat = feat.view(feat.size(0), -1)
        value = F.relu(self.fc_value_hidden(feat))
        value = self.fc_value(value).view(-1, 1, self.atoms)
        advantage = F.relu(self.fc_advantage_hidden(feat))
        advantage = self.fc_advantage(advantage).view(-1, self.action_size, self.atoms)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        if log:
            return F.log_softmax(q_atoms, dim=2)
        else:
            return F.softmax(q_atoms, dim=2)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def get_action(self, state):
        with torch.no_grad():
            probs = self.forward(state)
            support = self.support.to(state.device)
            q_values = (probs * support.unsqueeze(0).unsqueeze(0)).sum(2)
            return q_values.argmax(1).item()
