import math
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# CHANGES
# reward scaled by length. Staying alive longer better
# scale step panalty

# Action mapping
ACTION_MAP = {
    0: (-1, 0),   # up
    1: (0, 1),    # right
    2: (1, 0),    # down
    3: (0, -1)    # left
}

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, board_size=14, render_mode=None):
        super().__init__()
        self.board_size = board_size
        self.render_mode = render_mode

        # Observation: 14x14x4 one-hot -> flattened
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(board_size * board_size * 4,),
            dtype=np.float32
        )

        # Actions: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)

        # Internal state
        self.snake = None
        self.food = None
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize snake in center
        
        self.snake = [(self.board_size // 3, self.board_size // 2)]
        self.food = (2*self.board_size // 3, self.board_size // 2)

        self.steps = 0
        self.steps_food = 0
        self.done = False
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = int(action)

        self.steps += 1 
        self.steps_food += 1
        if self.steps > self.board_size * 1.2:
            return self._get_obs(), -15.0, True, False, {}

        dx, dy = ACTION_MAP[action]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        # Check collisions
        if (not (0 <= new_head[0] < self.board_size and 0 <= new_head[1] < self.board_size)) \
            or new_head in self.snake:
            self.done = True
            return self._get_obs(), -20.0, True, False, {}

        # Move snake
        self.snake.insert(0, new_head)

        # Check food
        #print(new_head, self.food)

        if new_head == self.food:
            self._place_food()
            reward = 10.0 * len(self.snake) * 0.1

            # Win condition: snake fills the board
            if len(self.snake) == self.board_size * self.board_size:
                self.done = True
                reward = 100.0  # give a big reward for winning
                return self._get_obs(), reward, True, False, {}
        else:
            self.snake.pop()    


        if len(self.snake) > 1:
            pos_new, pos_old = self.snake[:2]
            if math.dist(pos_new, self.food) < math.dist(pos_old, self.food):
                reward = 0.05  # reward for moving closer
            else:
                reward = -0.002 if self.steps_food > len(self.snake) * 1.2 else 0
        
        obs = self._get_obs()
        return obs, reward, self.done, False, {}
    
    def food_distance_reward(self) -> float:
        pos_new, pos_old = self.snake[:2]

        if math.dist(pos_new, self.food) < math.dist(pos_old, self.food):
            return 0.05  # reward for moving closer
        
        step_panelty = -0.002 * 2/len(self.snake)

        reward = step_panelty if self.steps_food > len(self.snake) * 1.2 else 0

        return reward

    def _get_obs(self):
        board = np.zeros((self.board_size, self.board_size, 4), dtype=np.float32)
        for x, y in self.snake[1:]:
            board[y, x, 2] = 1.0  # body
        head_x, head_y = self.snake[0]
        board[head_y, head_x, 1] = 1.0  # head
        food_x, food_y = self.food
        board[food_y, food_x, 3] = 1.0  # food
        return board.flatten()

    def _place_food(self):
        free_cells = [(x, y) for x in range(self.board_size) for y in range(self.board_size) if (x, y) not in self.snake]
        self.food = tuple(self.np_random.choice(free_cells))
        self.steps_food = 0

    def render(self):
        if self.render_mode == "human":
            grid = np.full((self.board_size, self.board_size), ".")
            for x, y in self.snake[1:]:
                grid[y, x] = "o"
            head_x, head_y = self.snake[0]
            grid[head_y, head_x] = "H"
            fx, fy = self.food
            grid[fy, fx] = "F"
            print("\n".join(" ".join(row) for row in grid))
            print()
