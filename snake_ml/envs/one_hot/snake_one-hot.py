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

        # Observation: 14x14x4 one-hot -> flattened
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=((board_size * board_size * 4) ,),
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
        self.head_new_food = self.snake[0]
        self.inputs_since_food = 0
        self.steps_since_food = 0

        obs = self._get_obs()
        return obs, {"steps_since_food": self.steps_since_food}

    def step(self, action):
        action = int(action)

        self.steps_since_food += 1
        # max_steps_since_food = self.board_size**2 * 2
        # if self.steps_since_food > max_steps_since_food:
        #     return self._get_obs(), -2.0, True, False, { "length": len(self.snake), "cause": "Too many steps_since_food"}

        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        if action != opposites[self.direction]:
            self.direction = action
            self.inputs_since_food += 1

        dx, dy = ACTION_MAP[self.direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        # Check collisions
        if (not (0 <= new_head[0] < self.board_size and 0 <= new_head[1] < self.board_size)) \
            or new_head in self.snake:
            return self._get_obs(), *self.rw.deathPenalty(self, new_head)

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
        #         reward -= max_steps_since_food/(max_steps_since_food - self.steps_since_food + 1) - 1 # small penalty otherwise  

        return self._get_obs(), *self.rw.alive(self, new_head)
    

    def _place_food(self):
        free_cells = [(x, y) for x in range(self.board_size) for y in range(self.board_size) if (x, y) not in self.snake]
        self.food = tuple(self.np_random.choice(free_cells))
        self.head_new_food = self.snake[0]
        self.inputs_since_food = 0
        self.steps_since_food = 0


    def _get_obs(self):
        board = np.zeros((self.board_size, self.board_size, 4), dtype=np.float32)

        # Snake head
        head_x, head_y = self.snake[0]
        board[head_y, head_x, 1] = 1.0
        
        # Snake body
        for x, y in self.snake[1:]:
            board[y, x, 2] = 1.0

        # Food
        food_x, food_y = self.food
        board[food_y, food_x, 3] = 1.0

        # Flatten the board
        flat_board = board.flatten()

        return flat_board

    def render(self):
        render(self.snake, self.food, self.board_size, self.steps_since_food)
        