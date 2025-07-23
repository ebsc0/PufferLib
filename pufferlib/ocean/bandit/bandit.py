'''C-based multi-armed bandit environment.'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.bandit import binding

class Bandit(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, k=10, drift=0.01, render_mode=None, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(k)
        self.render_mode = render_mode
        self.num_agents = num_envs
        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed,
            k=k, drift=drift, history_size=512)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        info = [binding.vec_log(self.c_envs)]
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

# Simple neural network for agents
import torch
from torch import nn
import torch.nn.functional as F
from pufferlib.pytorch import layer_init

class Policy(nn.Module):
    def __init__(self, env, hidden_size=16):
        super().__init__()
        self.is_continuous = False
        obs = int(np.prod(env.single_observation_space.shape))
        self.fc = layer_init(nn.Linear(obs, hidden_size))
        self.actor = layer_init(nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value = layer_init(nn.Linear(hidden_size, 1), std=1)

    def encode_observations(self, x, state=None):
        x = x.view(x.shape[0], -1).float()
        return F.relu(self.fc(x))

    def decode_actions(self, hidden, state=None):
        return self.actor(hidden), self.value(hidden)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        return self.decode_actions(hidden)
