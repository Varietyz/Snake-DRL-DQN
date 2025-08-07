# snake_game/core/food.py
import random

def spawn_food(snake_body, grid_width, grid_height):
    while True:
        pos = (
            random.randint(0, grid_width - 1),
            random.randint(0, grid_height - 1)
        )
        if pos not in snake_body:
            return pos

def should_spawn_bonus(chance):
    return random.random() < chance

class Food:
    def __init__(self, grid_width, grid_height):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.position = (0, 0)

    def spawn(self, snake_bodies):
        flattened = [segment for body in snake_bodies for segment in body]
        self.position = spawn_food(flattened, self.grid_width, self.grid_height)
