# ui/ai_controller.py
try:
    from ai.dqn_agent import DQNAgent
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("AI module not available. Install numpy to enable AI features.")

class AIController:
    def __init__(self, enabled, snake, food, bonus_food, grid_width, grid_height):
        self.enabled = enabled and AI_AVAILABLE
        self.snake = snake
        self.food = food
        self.bonus_food = bonus_food
        self.grid_width = grid_width
        self.grid_height = grid_height

        self.agent = DQNAgent() if self.enabled else None

        self.prev_state = None
        self.prev_action = None
        self.update_target_counter = 0

    def reset(self, snake, food, bonus_food):
        self.snake = snake
        self.food = food
        self.bonus_food = bonus_food
        self.prev_state = None
        self.prev_action = None
        self.update_target_counter = 0

    def get_action(self):
        if not self.enabled or not self.agent:
            return None

        current_state = self.agent.get_state(self.snake, self.food, self.bonus_food, self.grid_width, self.grid_height)
        action = self.agent.act(current_state)

        # Learn from previous state/action
        if self.prev_state is not None:
            reward = self.calculate_reward(current_state)
            self.agent.remember(self.prev_state, self.prev_action, reward, current_state, False)

        self.prev_state = current_state
        self.prev_action = action

        return action

    def calculate_reward(self, current_state):
        head = self.snake.get_head()
        score = self.agent.score if hasattr(self.agent, 'score') else 0  # fallback
        multiplier = 1 + score

        # Terminal state penalty
        if (not (0 <= head[0] < self.grid_width) or not (0 <= head[1] < self.grid_height) or
                self.snake.hits_self()):
            return -750 * multiplier

        if self.bonus_food and head == self.bonus_food:
            return 300 * multiplier

        if head == self.food:
            return 100 * multiplier

        # Reward shaping based on distance to food/bonus food
        dist_food_now = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        dist_food_prev = abs(self.prev_state[0][7] - self.food[0]) + abs(self.prev_state[0][8] - self.food[1])

        food_reward = 2.0 * multiplier if dist_food_now < dist_food_prev else -0.2 * multiplier

        bonus_reward = 0
        if self.bonus_food:
            dist_bonus_now = abs(head[0] - self.bonus_food[0]) + abs(head[1] - self.bonus_food[1])
            dist_bonus_prev = abs(self.prev_state[0][7] - self.bonus_food[0]) + abs(self.prev_state[0][8] - self.bonus_food[1])
            bonus_reward = 0.5 * multiplier if dist_bonus_now < dist_bonus_prev else -0.1 * multiplier

        return food_reward + bonus_reward

    def replay_and_update(self):
        if self.enabled and self.agent:
            self.agent.replay()
            self.update_target_counter += 1
            if self.update_target_counter % 100 == 0:
                self.agent.update_target_network()
                self.agent.save_model()

    def final_reward_on_death(self):
        if self.enabled and self.agent and self.prev_state is not None:
            final_reward = -100
            self.agent.remember(self.prev_state, self.prev_action, final_reward, self.prev_state, True)
            self.agent.replay()
            self.agent.save_model()