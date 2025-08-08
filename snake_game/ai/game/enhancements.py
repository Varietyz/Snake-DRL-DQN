# ai/game/enhancements.py

def apply_enhancement(agent_state, enhancement_type):
    if enhancement_type == "speed_boost":
        agent_state["learning_rate"] *= 1.2
        agent_state["exploration_rate"] *= 0.8
    elif enhancement_type == "double_reward":
        agent_state["reward_multiplier"] = 2.0
    elif enhancement_type == "experience_buffer_boost":
        agent_state["memory_size"] += 100

    return agent_state
