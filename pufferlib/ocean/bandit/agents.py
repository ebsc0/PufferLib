import torch
from torch import nn
import pufferlib

class LinearAgent(nn.Module):
    """Simple linear policy for the bandit."""
    def __init__(self, env: pufferlib.PufferEnv):
        super().__init__()
        n = env.single_action_space.n
        self.bias = nn.Parameter(torch.zeros(n))
        self.is_continuous = False

    def forward(self, observations, state=None):
        batch = observations.shape[0]
        logits = self.bias.expand(batch, -1)
        value = torch.zeros(batch, 1, device=logits.device)
        return logits, value

class MLPAgent(nn.Module):
    """Two layer MLP policy for the bandit."""
    def __init__(self, env: pufferlib.PufferEnv, hidden_size: int = 16):
        super().__init__()
        n = env.single_action_space.n
        self.is_continuous = False
        self.net = nn.Sequential(
            nn.Linear(env.single_observation_space.shape[0], hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n),
        )
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, observations, state=None):
        hidden = self.net[0](observations)
        hidden = self.net[1](hidden)
        logits = self.net[2](hidden)
        value = self.value(hidden)
        return logits, value
