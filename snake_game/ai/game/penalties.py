# ai/game/penalties.py

def apply_penalty(reward: float, game_info: dict) -> float:
    penalties = {
        "collision": -1000.0,
        "timeout": -100.0,
        "invalid_move": -500.0,
    }

    violation = game_info.get("violation")
    if violation in penalties:
        reward += penalties[violation]  # additive penalty
    return reward

