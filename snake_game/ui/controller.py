# ui/controller.py
from config import (GRID_WIDTH, GRID_HEIGHT, SPEED, SPEED_INCREMENT, MIN_SPEED,
                    BONUS_FOOD_CHANCE, BONUS_FOOD_POINTS, STATE_SIZE, ACTION_SIZE)
from core.snake import Snake
from core.food import spawn_food, should_spawn_bonus
from core.score import ScoreManager
from ui.renderer import Renderer
from ui.input_handler import InputHandler
from ui.bonus_manager import BonusFoodManager

from ai.agent_plugin import AgentPlugin
from ai.connector import Connector

import time

class GameController:
    def __init__(self, root, canvas):
        self.root = root
        self.canvas = canvas
        self.renderer = Renderer(canvas)
        self.score = ScoreManager()
        
        self.games_played = 0
        self.time_playing = 0.0  # seconds
        self._last_update_time = time.time()
        
        self._init_game_entities()

        # Setup bonus food manager
        self.bonus_manager = BonusFoodManager(self.root, self.snake.body, self.food, spawn_food)

        # Setup input handler without AI toggle
        self.input_handler = InputHandler(
            root,
            self.snake,
        )

        self.agent = AgentPlugin(state_size=STATE_SIZE, action_size=ACTION_SIZE)

        self.connector = Connector(self, self.agent)
        
        self.current_speed = SPEED
        self.running = True

        self._game_loop()

    def _init_game_entities(self):
        self.snake = Snake((5, 5))
        self.food = spawn_food(self.snake.body, GRID_WIDTH, GRID_HEIGHT)
        self.score.reset()

    def _increase_speed(self):
        new_speed = self.current_speed - SPEED_INCREMENT
        if new_speed < MIN_SPEED:
            new_speed = MIN_SPEED
        self.current_speed = new_speed

    def _end_game(self):
        self.running = False
        self.bonus_manager.remove_bonus_food()
        self.canvas.delete("all")
        self._start_new_game()

        
    def _start_new_game(self):
        self.games_played += 1
        self._init_game_entities()
        self.bonus_manager = BonusFoodManager(self.root, self.snake.body, self.food, spawn_food)
        self.running = True
        self.current_speed = SPEED
        self._last_update_time = time.time() 
        self._game_loop()

    def _game_loop(self):
        if not self.running:
            return
        
        current_time = time.time()
        delta = current_time - self._last_update_time
        self._last_update_time = current_time
        self.update_time(delta)  # update time_playing
        
        self.connector.update()

        self.snake.move()
        head = self.snake.get_head()
        prev_head = self.snake.get_previous_head()
        prev_food_dist = abs(prev_head[0] - self.food[0]) + abs(prev_head[1] - self.food[1])
        curr_food_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        delta_dist = prev_food_dist - curr_food_dist

        # Check collisions
        game_over = False
        if not (0 <= head[0] < GRID_WIDTH) or not (0 <= head[1] < GRID_HEIGHT) or self.snake.hits_self():
            game_over = True

        # Check food collision
        if head == self.food:
            self.snake.grow()
            self.score.increment()
            self._increase_speed()
            self.food = spawn_food(self.snake.body, GRID_WIDTH, GRID_HEIGHT)

            if not self.bonus_manager.bonus_food and should_spawn_bonus(BONUS_FOOD_CHANCE):
                self.bonus_manager.spawn_bonus_food()

        # Check bonus food collision
        if self.bonus_manager.bonus_food and head == self.bonus_manager.bonus_food:
            self.snake.grow()
            self.score.increment(BONUS_FOOD_POINTS)
            self.bonus_manager.remove_bonus_food()

        game_info = {
            "ate_food": head == self.food,
            "ate_bonus_food": self.bonus_manager.bonus_food and head == self.bonus_manager.bonus_food,
            "collision": game_over,
            "distance_to_food_delta": delta_dist,
            "violation": "collision" if game_over else None,
        }

        self.renderer.draw_from_controller(self)

        # Signal to agent whether game is done
        self.agent.remember_and_train(self, game_info, game_over)

        if game_over:
            self._end_game()
            return

        self.root.after(self.current_speed, self._game_loop)
        
    def update_time(self, delta_seconds):
        self.time_playing += delta_seconds
