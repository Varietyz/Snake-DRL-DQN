from config import GRID_SIZE, COLORS
from ui.metrics import RenderMetrics

class Renderer:
    def __init__(self, canvas):
        self.canvas = canvas

    def draw(self, snake_body, food_pos, score=None, high_score=None,
             bonus_food_pos=None, bonus_food_visible=True,
             current_speed=None, snake_length=None, games_played=None, time_playing=None):
        self.canvas.delete("all")
        for x, y in snake_body:
            self._draw_cell(x, y, COLORS["snake"])
        fx, fy = food_pos
        self._draw_cell(fx, fy, COLORS["food"])
        if bonus_food_pos and bonus_food_visible:
            bx, by = bonus_food_pos
            self._draw_cell(bx, by, COLORS["bonus_food"])
        if score is not None:
            self._draw_score(score, high_score, current_speed, snake_length, games_played, time_playing)

    def draw_from_controller(self, controller):
        """
        Helper method to extract metrics from GameController and draw accordingly.
        """
        metrics = RenderMetrics(
            current_score=controller.score.current_score,
            high_score=controller.score.high_score,
            bonus_food_pos=controller.bonus_manager.bonus_food,
            bonus_food_visible=controller.bonus_manager.bonus_visible,
            current_speed=controller.current_speed,
            snake_length=len(controller.snake.body),
            # Include games_played if you have added it in controller
            games_played=getattr(controller, 'games_played', None),
            time_playing=getattr(controller, 'time_playing', None),
        )

        self.draw(
            snake_body=controller.snake.body,
            food_pos=controller.food,
            score=metrics.current_score,
            high_score=metrics.high_score,
            bonus_food_pos=metrics.bonus_food_pos,
            bonus_food_visible=metrics.bonus_food_visible,
            current_speed=metrics.current_speed,
            snake_length=metrics.snake_length,
            games_played=metrics.games_played,
            time_playing=metrics.time_playing,
        )

    def _draw_cell(self, x, y, color):
        x1 = x * GRID_SIZE
        y1 = y * GRID_SIZE
        x2 = x1 + GRID_SIZE
        y2 = y1 + GRID_SIZE
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def _draw_score(self, score, high_score, current_speed=None, snake_length=None, games_played=None, time_playing=None):
        # Format time_playing as mm:ss
        time_text = ""
        if time_playing is not None:
            minutes = int(time_playing // 60)
            seconds = int(time_playing % 60)
            time_text = f"Time: {minutes:02d}:{seconds:02d}"

        text = f"Score: {score} | High Score: {high_score} | "
        if current_speed is not None:
            text += f"Speed: {current_speed} | "
        if snake_length is not None:
            text += f"Length: {snake_length} | "
        if games_played is not None:
            text += f"Games Played: {games_played} | "
        text += time_text

        self.canvas.create_text(
            10, 10,
            anchor="nw",
            fill="white",
            font="Arial 10",
            text=text
        )
