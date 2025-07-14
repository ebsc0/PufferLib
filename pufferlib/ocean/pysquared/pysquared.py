'''Pure python version of Squared, a simple single-agent sample environment.
   Use this as a template for your own envs.
'''

# We only use Gymnasium for their spaces API for compatibility with other libraries.
# Gymnasium is the standard API for reinforcement learning environments in Python.
import gymnasium
# NumPy is used for numerical operations, especially for handling the observation grid.
import numpy as np

# PufferLib is the core library we are using.
import pufferlib

# --- Action and Observation Constants ---
# These constants make the code more readable by giving names to the numerical values
# that represent actions and objects in the environment.

# Define the integer values for each possible action.
NOOP = 0  # Do nothing
DOWN = 1  # Move down
UP = 2    # Move up
LEFT = 3  # Move left
RIGHT = 4 # Move right

# Define the integer values for what can be in a grid cell.
EMPTY = 0  # An empty cell
AGENT = 1  # The cell occupied by the agent
TARGET = 2 # The cell occupied by the target

# --- Environment Class Definition ---
# Inherit from pufferlib.PufferEnv, which provides the core PufferLib functionality
# for vectorized environments and in-place data handling.
class PySquared(pufferlib.PufferEnv):
    '''
    A simple grid-world environment where an agent must navigate to a target.
    This class demonstrates the basic structure of a PufferLib environment.
    '''
    # The __init__ method is the constructor for the class.
    # It is called when a new instance of the environment is created.
    # It requires keyword arguments: render_mode, buf, and seed, which are standard for PufferEnv.
    def __init__(self, render_mode='ansi', size=11, buf=None, seed=0):
        '''
        Initializes the environment.

        Args:
            render_mode (str): The mode for rendering. 'ansi' for text-based rendering.
            size (int): The height and width of the square grid.
            buf (pufferlib.Buffer): The buffer object provided by PufferLib's vectorization
                                    for in-place data storage. This is handled automatically.
            seed (int): The random seed for reproducibility.
        '''
        # --- Required PufferEnv Attributes ---
        # These attributes must be defined BEFORE calling the super().__init__ method.

        # single_observation_space: Defines the observation space for a single agent.
        # It's a 1D vector representing the flattened grid.
        # gymnasium.spaces.Box is used for continuous or multi-dimensional data.
        # Here, it represents a grid of size*size, with each cell having a value from 0 to 2.
        # Note: The original code had high=1, which is incorrect since TARGET is 2. It should be high=2.
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(size*size,), dtype=np.uint8)

        # single_action_space: Defines the action space for a single agent.
        # gymnasium.spaces.Discrete(5) means there are 5 possible integer actions (0 to 4).
        self.single_action_space = gymnasium.spaces.Discrete(5)

        # render_mode: Stores the rendering mode.
        self.render_mode = render_mode

        # num_agents: The number of agents in the environment. For single-agent, this is 1.
        self.num_agents = 1

        # Call the parent class's constructor. This is a crucial step.
        # It sets up the buffer (`self.buf`) and related attributes like
        # `self.observations`, `self.rewards`, etc., which are views into the shared buffer.
        super().__init__(buf)

        # --- Custom Environment Attributes ---
        # You can add any other attributes you need for your environment's logic.
        self.size = size

    # The reset method is called to begin a new episode.
    def reset(self, seed=0):
        '''
        Resets the environment to a starting state.

        Args:
            seed (int): An optional seed for this specific reset.

        Returns:
            observations: The initial state of the environment.
            infos: An empty list, as there's no extra info at reset.
        '''
        # Clear the entire observation grid for the first agent (index 0).
        # self.observations is a NumPy array view into the PufferLib buffer.
        self.observations[0, :] = EMPTY

        # Place the agent in the center of the grid.
        self.observations[0, self.size*self.size//2] = AGENT

        # Store the agent's row and column for easy access.
        self.r = self.size//2
        self.c = self.size//2

        # Reset the tick counter for the episode length.
        self.tick = 0

        # Randomly place the target in a location that is not the agent's starting position.
        while True:
            target_r, target_c = np.random.randint(0, self.size, 2)
            if target_r != self.r or target_c != self.c:
                self.observations[0, target_r*self.size + target_c] = TARGET
                break

        # Return the initial observation from the buffer and an empty info list.
        # No need to create a new copy; PufferLib works with the buffer directly.
        return self.observations, []

    # The step method processes a single timestep of the environment.
    def step(self, actions):
        '''
        Executes one time step in the environment.

        Args:
            actions (list/np.array): The action(s) to be taken by the agent(s).

        Returns:
            observations: The observation after the action is taken.
            rewards: The reward received after the action.
            terminals: A boolean indicating if the episode has ended.
            truncations: A boolean indicating if the episode was truncated (e.g., time limit).
            infos: A list of dictionaries with auxiliary diagnostic information.
        '''
        # Get the action for the first agent. `actions` is an array.
        atn = actions[0]

        # --- In-place Updates ---
        # It's critical to reset terminal and reward at the start of the step,
        # as the values in the buffer persist from the previous step.
        self.terminals[0] = False
        self.rewards[0] = 0

        # Erase the agent from its previous position on the grid.
        self.observations[0, self.r*self.size + self.c] = EMPTY

        # Update the agent's row and column based on the chosen action.
        if atn == DOWN:
            self.r += 1
        elif atn == RIGHT:
            self.c += 1
        elif atn == UP:
            self.r -= 1
        elif atn == LEFT:
            self.c -= 1
        # If atn is NOOP, the position doesn't change.

        # Initialize the info list for this step.
        info = []
        # Calculate the new 1D position in the flattened grid.
        pos = self.r*self.size + self.c

        # --- Check for Termination Conditions ---
        # Check if the episode should end due to timeout or going out of bounds.
        if (self.tick > 3*self.size  # Timeout
                or self.r < 0 or self.c < 0 # Out of bounds
                or self.r >= self.size or self.c >= self.size):
            self.terminals[0] = True
            self.rewards[0] = -1.0
            info = [{'reward': -1.0}]
            self.reset() # Reset the environment for the next episode.
        # Check if the agent has reached the target.
        elif self.observations[0, pos] == TARGET:
            self.terminals[0] = True
            self.rewards[0] = 1.0
            info = [{'reward': 1.0}]
            self.reset() # Reset the environment for the next episode.
        # If the episode has not ended, place the agent in the new position.
        else:
            self.observations[0, pos] = AGENT
            self.tick += 1

        # Return the views into the buffer. PufferLib handles the data.
        return self.observations, self.rewards, self.terminals, self.truncations, info

    # The render method provides a visualization of the environment.
    def render(self):
        '''
        Renders the current state of the environment.
        This implementation uses ANSI escape codes for console-based coloring.
        '''
        chars = []
        # Reshape the 1D observation vector into a 2D grid for rendering.
        grid = self.observations.reshape(self.size, self.size)
        for row in grid:
            for val in row:
                if val == AGENT:
                    color = 94  # Blue
                elif val == TARGET:
                    color = 91  # Red
                else:
                    color = 90  # Grey
                # Append the colored block character.
                chars.append(f'\033[{color}m██\033[0m')
            chars.append('\n')
        return ''.join(chars)

    # The close method is for any necessary cleanup.
    def close(self):
        '''
        Performs any necessary cleanup before the environment is destroyed.
        '''
        pass

# This block of code runs only when the script is executed directly.
# It's useful for testing and benchmarking the environment.
if __name__ == '__main__':
    # Create an instance of the environment.
    env = PySquared()
    # Reset the environment to get the initial state.
    env.reset()
    steps = 0

    # Pre-generate a cache of random actions to avoid that being a bottleneck during timing.
    CACHE = 1024
    actions = np.random.randint(0, 5, (CACHE, 1))

    import time
    start = time.time()
    # Run the environment for 10 seconds.
    while time.time() - start < 10:
        # Step the environment with a random action from the cache.
        env.step(actions[steps % CACHE])
        steps += 1

    # Calculate and print the performance in Steps Per Second (SPS).
    print('PySquared SPS:', int(steps / (time.time() - start)))