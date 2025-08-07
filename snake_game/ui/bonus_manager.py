# ui/bonus_manager.py
from config import (GRID_WIDTH, GRID_HEIGHT, BONUS_FOOD_DURATION, BONUS_FOOD_FLICKER_START)

class BonusFoodManager:
    def __init__(self, root, snake_body, food, spawn_func):
        self.root = root
        self.snake_body = snake_body
        self.food = food
        self.spawn_food = spawn_func

        self.bonus_food = None
        self.bonus_visible = True
        self.bonus_timeout_id = None
        self.bonus_flicker_id = None

    def spawn_bonus_food(self):
        self.bonus_food = self.spawn_food(list(self.snake_body) + [self.food], GRID_WIDTH, GRID_HEIGHT)
        self.bonus_visible = True
        self.bonus_timeout_id = self.root.after(BONUS_FOOD_DURATION, self.remove_bonus_food)

        flicker_delay = BONUS_FOOD_DURATION - BONUS_FOOD_FLICKER_START
        self.root.after(flicker_delay, self._start_bonus_flicker)

    def _start_bonus_flicker(self):
        if self.bonus_food:
            self._toggle_bonus_visibility()

    def _toggle_bonus_visibility(self):
        if self.bonus_food:
            self.bonus_visible = not self.bonus_visible
            self.bonus_flicker_id = self.root.after(200, self._toggle_bonus_visibility)

    def remove_bonus_food(self):
        self.bonus_food = None
        self.bonus_visible = True
        if self.bonus_timeout_id:
            self.root.after_cancel(self.bonus_timeout_id)
            self.bonus_timeout_id = None
        if self.bonus_flicker_id:
            self.root.after_cancel(self.bonus_flicker_id)
            self.bonus_flicker_id = None