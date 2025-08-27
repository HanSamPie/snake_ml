import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from snake_ml.render import render

# CHANGES
# truncate after board_size * 1.2 steps
# reward distance reduction by 0.05 otherwise -0.002

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
        self.direction = 1 # right

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize snake in center
        
        self.snake = [(self.board_size // 3, self.board_size // 2)]
        self.food = (2*self.board_size // 3, self.board_size // 2)

        self.steps = 0
        self.done = False
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = int(action)

        reward = 0.0

        self.steps += 1
        if self.steps > self.board_size**2 * 1.2:
            return self._get_obs(), -10.0, True, False, { "length": len(self.snake), "cause": "Too many steps"}

        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        if action != opposites[self.direction]:
            self.direction = action

        dx, dy = ACTION_MAP[self.direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        # Check collisions
        if (not (0 <= new_head[0] < self.board_size and 0 <= new_head[1] < self.board_size)) \
            or new_head in self.snake:
            self.done = True
            return self._get_obs(), -20.0, True, False, { "length": len(self.snake), "cause": "collision"}

        # Move snake
        self.snake.insert(0, new_head)

        # Check food
        if new_head == self.food:
            self._place_food()
            reward = 10.0

            # Win condition: snake fills the board
            if len(self.snake) == self.board_size * self.board_size:
                self.done = True
                reward = 100.0  # give a big reward for winning
                return self._get_obs(), reward, True, False, { "length": len(self.snake), "cause": "win"}
        else:
            self.snake.pop()    


        obs = self._get_obs()
        pos_new, pos_old = self.snake[:2]

        if math.dist(pos_new, self.food) < math.dist(pos_old, self.food):
            reward += 0.05  # reward for moving closer
        reward += -0.002   # small penalty otherwise

        return obs, reward, self.done, False, { "length": len(self.snake), "cause": "EoF"}

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

    def render(self):
        render(self.snake, self.food, self.board_size)
