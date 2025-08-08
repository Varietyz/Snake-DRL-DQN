# ai/game/penalties.py

def apply_penalty(reward: float, game_info: dict) -> float:
    penalties = {
        "collision": -100.0,
        "timeout": -10.0,
        "invalid_move": -5.0,
    }

    violation = game_info.get("violation")
    if violation in penalties:
        reward += penalties[violation]  # additive penalty
    return reward

