# config.py
GRID_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 20
INITIAL_SNAKE_LENGTH = 3
SPEED = 150  # in ms
SPEED_INCREMENT = 5  # ms to decrease per food eaten
MIN_SPEED = 50  # minimum speed (fastest)

# Bonus food settings
BONUS_FOOD_CHANCE = 0.15  # 15% chance to spawn bonus food
BONUS_FOOD_DURATION = 8000  # ms before bonus food disappears
BONUS_FOOD_POINTS = 5  # points for eating bonus food
BONUS_FOOD_FLICKER_START = 3000  # ms before disappear when flickering starts

# AI settings
AI_SPEED = 30  # ms - much faster than human play
AI_ENABLED_BY_DEFAULT = True
AI_LEARNING_RATE = 0.002
AI_EPSILON_DECAY = 0.995
AI_MIN_EPSILON = 0.01
AI_MEMORY_SIZE = 100000
AI_BATCH_SIZE = 32

COLORS = {
    "background": "#111",
    "snake": "#0f0",
    "food": "#f00",
    "bonus_food": "#ff0"  # yellow for bonus food
}