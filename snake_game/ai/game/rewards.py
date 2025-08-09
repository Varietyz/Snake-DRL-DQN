# ai/game/rewards.py
from config import GRID_WIDTH, GRID_HEIGHT
# Reward configuration — tweak these to adjust reward values easily
REWARDS_CONFIG = {
    "ate_food": 250.0,
    "ate_bonus_food": 1000.0,
    "distance_to_food_delta_positive": 15.0,
    "distance_to_food_delta_negative": 30.0,
    "survival": 0.8,
    "danger_penalty_per_unit": 3.0,
    "danger_avoidance_bonus": 15.0,
    "wall_proximity_weight": 1.5,
}

def calculate_reward(prev_state, action, current_state, info=None) -> float:
    score = info.get("score", 0) if info else 0
    reward = 0.0
    
    # Basic rewards
    if info and info.get("ate_food", False):
        reward += REWARDS_CONFIG["ate_food"]
    if info and info.get("ate_bonus_food", False):
        reward += REWARDS_CONFIG["ate_bonus_food"]
        
    # Distance-based rewards (more granular)
    if info and "distance_to_food_delta" in info:
        delta = info["distance_to_food_delta"]
        if delta > 0:  # moving closer
            reward += REWARDS_CONFIG["distance_to_food_delta_positive"] * delta
        else:  # moving away
            reward += REWARDS_CONFIG["distance_to_food_delta_negative"] * delta  # smaller penalty
            
    reward += score * reward
    
    # Survival reward (encourage staying alive)
    reward += REWARDS_CONFIG["survival"]
    
    # Danger avoidance (extracted from state vector)
    if prev_state is not None:
        prev_dangers = prev_state[-4:]
        current_dangers = current_state[-4:]
        
        # Penalize moving toward danger
        action_danger = current_dangers[action]
        reward -= REWARDS_CONFIG["danger_penalty_per_unit"] * action_danger
        
        # Reward avoiding immediate danger
        if any(d > 0.5 for d in prev_dangers) and all(d < 0.5 for d in current_dangers):
            reward += REWARDS_CONFIG["danger_avoidance_bonus"]
    
    # Wall proximity awareness
    head_x, head_y = current_state[100], current_state[101]
    grid_width, grid_height = GRID_WIDTH, GRID_HEIGHT

    
    # Encourage staying away from walls
    # Convert normalized head position back to tile units
    head_tile_x = int(head_x * grid_width)
    head_tile_y = int(head_y * grid_height)

    tiles_from_wall = min(
        head_tile_x,
        grid_width - 1 - head_tile_x,
        head_tile_y,
        grid_height - 1 - head_tile_y
    )

    # Normalize tile distance back to range 0.0 - 1.0 (optional for scaling)
    normalized_wall_proximity = tiles_from_wall / max(grid_width, grid_height)

    reward += REWARDS_CONFIG["wall_proximity_weight"] * normalized_wall_proximity

    return reward
