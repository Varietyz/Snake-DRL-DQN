# ui/renderer.py
from config import GRID_SIZE, COLORS

class Renderer:
    def __init__(self, canvas):
        self.canvas = canvas

    def draw(self, snake_body, food_pos, score=None, high_score=None, bonus_food_pos=None, bonus_food_visible=True, ai_mode=False):
        self.canvas.delete("all")
        for x, y in snake_body:
            self._draw_cell(x, y, COLORS["snake"])
        fx, fy = food_pos
        self._draw_cell(fx, fy, COLORS["food"])
        if bonus_food_pos and bonus_food_visible:
            bx, by = bonus_food_pos
            self._draw_cell(bx, by, COLORS["bonus_food"])
        if score is not None:
            self._draw_score(score, high_score, ai_mode)

    def _draw_cell(self, x, y, color):
        x1 = x * GRID_SIZE
        y1 = y * GRID_SIZE
        x2 = x1 + GRID_SIZE
        y2 = y1 + GRID_SIZE
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def _draw_score(self, score, high_score, ai_mode=False):
        mode_text = " [AI PLAYING]" if ai_mode else ""
        self.canvas.create_text(
            10, 10,
            anchor="nw",
            fill="white",
            font="Arial 10",
            text=f"Score: {score}  High Score: {high_score}{mode_text}"
        )