# ui/metrics.py
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class RenderMetrics:
    current_score: int
    high_score: int
    bonus_food_pos: Optional[Tuple[int, int]]
    bonus_food_visible: bool
    current_speed: int
    snake_length: int
    games_played: int  # new field
    time_playing: Optional[float] = None  # seconds elapsed
    
def extract_render_metrics(controller) -> RenderMetrics:
    return RenderMetrics(
        current_score=controller.score.current_score,
        high_score=controller.score.high_score,
        bonus_food_pos=controller.bonus_manager.bonus_food,
        bonus_food_visible=controller.bonus_manager.bonus_visible,
        current_speed=controller.current_speed,
        snake_length=len(controller.snake.body),
        games_played=controller.games_played,  # include here
        time_playing=getattr(controller, "time_playing", None),  # new field
    )

