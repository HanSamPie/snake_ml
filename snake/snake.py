import math

from collections import deque
import random

import numpy as np

# 0=up, 1=right, 2=down, 3=left
ACTION_MAP = {
    0: (-1, 0), # up
    1: (0, 1), # right
    2: (1, 0), #down
    3: (0, -1) # left
}

OPPOSITE_MAP = {
    0: (1, 0), # up
    1: (0, -1), # right
    2: (-1, 0), #down
    3: (0, 1) # left
}

class SnakeGame:
    def __init__(self, board_size=14):
        self.board_size = board_size
        self.max_length = board_size * board_size
        self.reset()

    def reset(self):
        self.snake = deque()
        self.snake.append((self.board_size//3, self.board_size//2))
        self.food = (2 * self.board_size//3, self.board_size//2)
        self.direction = (0, 1)  # moving right initially
        self.score = 0

    def _spawn_food(self):
        snake_set = set(self.snake)
        while True:
            food = (random.randint(0, self.board_size-1), random.randint(0, self.board_size-1))
            if food not in snake_set:
                self.food = food
                break

    def step(self, action):
        if len(self.snake) == self.max_length:
            # state, reward, terminated, truncated, info
            return self._get_obs(), 10.0, True, False, {"won": True}
        
        
        # check for opposite direction
        if ACTION_MAP[action] != OPPOSITE_MAP[action]:
            self.direction = ACTION_MAP[action]
        else:
            pass

        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])

        # check collision
        if new_head in self.snake:
            return self._get_obs(), -1.0, True, False, {"collision": True}

        x, y = new_head

        if new_head == self.food:
            self._spawn_food()

            return self._get_obs(), 1.0, False, False, {"ate_food": True}
        # out of board
        elif x < 0 or y < 0 or x >= self.board_size or y >= self.board_size:
            return self._get_obs(), -1.0, True, False, {"wall": True}
        else:
            self.snake.pop()
        
        # move
        self.snake.appendleft(new_head)

    def _get_obs(self):
         # convert game board to 14x14x4 one-hot and flatten
        board = np.zeros((self.board_size, self.board_size, 4), dtype=np.float32)
        # mark snake head, body, food
        for x, y in self.snake[1:]:
            board[y, x, 2] = 1.0  # body
        head_x, head_y = self.snake[0]
        board[head_y, head_x, 1] = 1.0  # head
        food_x, food_y = self.food
        board[food_y, food_x, 3] = 1.0  # food
        return board.flatten()