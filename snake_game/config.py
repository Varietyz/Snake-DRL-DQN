# config.py
GRID_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 20

INITIAL_SNAKE_LENGTH = 3

SPEED = 10  # in ms
SPEED_INCREMENT = 1  # ms to decrease per food eaten
MIN_SPEED = 2  # minimum speed (fastest)

# Bonus food settings
BONUS_FOOD_CHANCE = 0.15  # 15% chance to spawn bonus food
BONUS_FOOD_DURATION = 8000  # ms before bonus food disappears
BONUS_FOOD_POINTS = 5  # points for eating bonus food
BONUS_FOOD_FLICKER_START = 3000  # ms before disappear when flickering starts

# AI settings
STATE_SIZE = 50 * 2 + 4 + 4 + 4 + 4 # snake segments + head position + direction + food positions + danger
ACTION_SIZE = 4  # up, right, down, left
STOP_TRAINING = False # whether to train the AI or just run it

COLORS = {
    "background": "#111",
    "snake": "#0f0",
    "food": "#f00",
    "bonus_food": "#ff0"  # yellow for bonus food
}