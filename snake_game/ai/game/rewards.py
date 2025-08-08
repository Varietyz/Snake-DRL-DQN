# ai/game/rewards.py

def calculate_reward(prev_state, action, current_state, info=None) -> float:
    reward = 0.0
    if info and info.get("ate_food", False):
        reward += 10.0
    if info and info.get("ate_bonus_food", False):
        reward += 20.0
    if info and "distance_to_food_delta" in info:
        reward += max(0, info["distance_to_food_delta"]) * 2

    if reward == 0.0:
        reward -= 2.0
        
    return reward
