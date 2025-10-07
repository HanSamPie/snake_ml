import numpy as np
from rich.console import Console
import time

def render(snake, food, board_size, steps):
    console = Console()
    
    grid = np.full((board_size, board_size), " ")
    for x, y in snake[1:]:
        grid[y, x] = "#"
    head_x, head_y = snake[0]
    grid[head_y, head_x] = "$"
    fx, fy = food
    grid[fy, fx] = "*"

    console.clear()
    console.print(grid, f"\nsteps since food: {steps}")
    time.sleep(0.05)