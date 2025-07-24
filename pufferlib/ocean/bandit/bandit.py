import numpy as np
import gymnasium
import torch
import torch.nn as nn

import pufferlib
from pufferlib.ocean.bandit import binding

class Bandit(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, num_arms=10, horizon=100, render_mode=None, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(num_arms)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.horizon = horizon

        super().__init__(buf)
        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed,
            num_arms=num_arms, horizon=horizon
        )

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

class Policy(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc = nn.Linear(env.single_observation_space.shape[0], 16)
        self.actor = nn.Linear(16, env.single_action_space.n)
        self.value = nn.Linear(16, 1)

    def forward_eval(self, observations, state=None):
        x = torch.relu(self.fc(observations.float().view(observations.shape[0], -1)))
        logits = self.actor(x)
        values = self.value(x)
        return logits, values

    def forward(self, obs, state=None):
        return self.forward_eval(obs, state)

class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy=Policy, input_size=16, hidden_size=16):
        super().__init__(env, policy(env), input_size, hidden_size)
