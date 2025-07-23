import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.bandit import binding

class Bandit(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, num_arms=4, render_mode=None, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(num_arms)
        self.render_mode = render_mode
        self.num_agents = num_envs

        super().__init__(buf)
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            num_arms=num_arms,
        )
        self.tick = 0

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.tick % 128 == 0:
            info.append(binding.vec_log(self.c_envs))
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)
