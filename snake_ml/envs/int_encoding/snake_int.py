import gymnasium as gym
from gymnasium import spaces
import numpy as np
from snake_ml.render import render
from snake_ml.envs.rewards import Rewards

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
        self.rw = Rewards()

        # Observation: 14x14 int -> flattened
        self.observation_space = spaces.Box(
            low=0.0, high=3.0,
            shape=(board_size * board_size,),
            dtype=np.float32
        )

        # Actions: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)

        # Internal state
        self.snake = None
        self.food = None
        self.direction = 1 # right

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize snake in center
        
        self.snake = [(self.board_size // 3, self.board_size // 2)]
        self.food = (2*self.board_size // 3, self.board_size // 2)
        self.steps = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = int(action)

        # self.steps += 1
        # max_steps = self.board_size**2 * 2
        # if self.steps > max_steps:
        #     return self._get_obs(), -2.0, True, False, { "length": len(self.snake), "cause": "Too many steps"}

        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        if action != opposites[self.direction]:
            self.direction = action

        dx, dy = ACTION_MAP[self.direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        reward = 0.0

        # Check collisions
        if (not (0 <= new_head[0] < self.board_size and 0 <= new_head[1] < self.board_size)) \
            or new_head in self.snake:
            return self._get_obs(), *self.rw.deathPenalty(self)

        # Move snake
        self.snake.insert(0, new_head)

        # Check food
        if new_head == self.food:
            reward = self.rw.winReward(self)

            # Only place food if agent didn't win
            if not len(self.snake) == self.board_size * self.board_size:
                reward = self.rw.foodReward(self)
                self._place_food()
        
            return self._get_obs(), *reward
        else:
            self.snake.pop()  

        # if len(self.snake) > 1:
        #     pos_new, pos_old = self.snake[:2]

        #     if math.dist(pos_new, self.food) < math.dist(pos_old, self.food):
        #         reward += 0.1 #* (14-math.dist(pos_new, self.food))/14  # reward for moving closer
        #     else:
        #         reward -= max_steps/(max_steps - self.steps + 1) - 1 # small penalty otherwise  

        return self._get_obs(), *self.rw.alive(self)
    

    def _place_food(self):
        free_cells = [(x, y) for x in range(self.board_size) for y in range(self.board_size) if (x, y) not in self.snake]
        self.food = tuple(self.np_random.choice(free_cells))
        self.steps = 0


    def _get_obs(self):
        board = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        for x, y in self.snake[1:]:
            board[y, x] = 1.0  # body
        
        head_x, head_y = self.snake[0]
        board[head_y, head_x] = 2.0  # head
        
        food_x, food_y = self.food
        board[food_y, food_x] = 3.0  # food
        
        return board.flatten()

    def render(self):
        render(self.snake, self.food, self.board_size)
        