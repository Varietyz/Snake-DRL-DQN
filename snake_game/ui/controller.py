from config import (GRID_WIDTH, GRID_HEIGHT, SPEED, SPEED_INCREMENT, MIN_SPEED,
                    BONUS_FOOD_CHANCE, BONUS_FOOD_POINTS, AI_SPEED, AI_ENABLED_BY_DEFAULT)
from core.snake import Snake
from core.food import spawn_food, should_spawn_bonus
from core.score import ScoreManager
from ui.renderer import Renderer
from ui.input_handler import InputHandler
from ui.ai_controller import AIController, AI_AVAILABLE
from ui.bonus_manager import BonusFoodManager

class GameController:
    def __init__(self, root, canvas):
        self.root = root
        self.canvas = canvas
        self.renderer = Renderer(canvas)
        self.score = ScoreManager()

        self.ai_enabled = AI_ENABLED_BY_DEFAULT and AI_AVAILABLE

        self._init_game_entities()

        # Setup AI controller
        self.ai_controller = AIController(self.ai_enabled, self.snake, self.food, None, GRID_WIDTH, GRID_HEIGHT)

        # Setup bonus food manager
        self.bonus_manager = BonusFoodManager(self.root, self.snake.body, self.food, spawn_food)

        # Setup input handler with callbacks
        self.input_handler = InputHandler(
            root,
            self.snake,
            ai_toggle_callback=self._toggle_ai,
            ai_enabled_getter=lambda: self.ai_enabled,
        )

        self.current_speed = AI_SPEED if self.ai_enabled else SPEED
        self.running = True

        self._game_loop()

    def _init_game_entities(self):
        self.snake = Snake((5, 5))
        self.food = spawn_food(self.snake.body, GRID_WIDTH, GRID_HEIGHT)
        self.score.reset()

    def _toggle_ai(self):
        if not AI_AVAILABLE:
            return
        self.ai_enabled = not self.ai_enabled
        self.current_speed = AI_SPEED if self.ai_enabled else SPEED
        self.ai_controller.enabled = self.ai_enabled
        if self.ai_enabled:
            print("AI enabled - Press SPACE to disable")
        else:
            print("AI disabled - Use arrow keys to play, SPACE to enable AI")

    def _increase_speed(self):
        new_speed = self.current_speed - SPEED_INCREMENT
        if new_speed < MIN_SPEED:
            new_speed = MIN_SPEED
        self.current_speed = new_speed

    def _end_game(self):
        self.running = False
        self.bonus_manager.remove_bonus_food()
        self.ai_controller.final_reward_on_death()

        countdown_time = 1 if self.ai_enabled else 5
        self._countdown(countdown_time)

    def _countdown(self, remaining):
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            fill="white",
            font="Arial 24 bold",
            text=f"Game Over\nRestarting in {remaining}..."
        )
        if remaining > 0:
            self.root.after(1000, self._countdown, remaining - 1)
        else:
            self._start_new_game()

    def _start_new_game(self):
        self._init_game_entities()
        self.bonus_manager = BonusFoodManager(self.root, self.snake.body, self.food, spawn_food)
        self.ai_controller.reset(self.snake, self.food, None)
        self.running = True
        self.current_speed = AI_SPEED if self.ai_enabled else SPEED
        self._game_loop()

    def _game_loop(self):
        if not self.running:
            return

        # AI decision making
        action = None
        if self.ai_enabled:
            action = self.ai_controller.get_action()
            if action is not None:
                directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
                if action < len(directions):
                    self.snake.set_direction(directions[action])

        self.snake.move()
        head = self.snake.get_head()

        # Check collisions
        if not (0 <= head[0] < GRID_WIDTH) or not (0 <= head[1] < GRID_HEIGHT) or self.snake.hits_self():
            self._end_game()
            return

        # Check food collision
        if head == self.food:
            self.snake.grow()
            self.score.increment()
            if not self.ai_enabled:
                self._increase_speed()
            self.food = spawn_food(self.snake.body, GRID_WIDTH, GRID_HEIGHT)

            if not self.bonus_manager.bonus_food and should_spawn_bonus(BONUS_FOOD_CHANCE):
                self.bonus_manager.spawn_bonus_food()

        # Check bonus food collision
        if self.bonus_manager.bonus_food and head == self.bonus_manager.bonus_food:
            self.snake.grow()
            self.score.increment(BONUS_FOOD_POINTS)
            self.bonus_manager.remove_bonus_food()

        # AI learning
        if self.ai_enabled:
            self.ai_controller.replay_and_update()

        # Update bonus food in AI controller
        self.ai_controller.bonus_food = self.bonus_manager.bonus_food
        self.ai_controller.food = self.food
        self.ai_controller.snake = self.snake

        self.renderer.draw(
            self.snake.body,
            self.food,
            self.score.current_score,
            self.score.high_score,
            self.bonus_manager.bonus_food,
            self.bonus_manager.bonus_visible,
            self.ai_enabled,
        )

        self.root.after(self.current_speed, self._game_loop)
