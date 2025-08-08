# ai/agent_plugin.py
import numpy as np
from ai.dqn_agent import DQNAgent
from ai.game.enhancements import apply_enhancement
from ai.game.rewards import calculate_reward
from ai.game.penalties import apply_penalty
from config import STOP_TRAINING

class AgentPlugin:
    def __init__(self, state_size: int, action_size: int):
        self.agent = DQNAgent(state_size=state_size, action_size=action_size)
        self.last_state = None
        self.last_action = None

    def preprocess_state(self, game_controller) -> np.ndarray:
        snake = game_controller.snake.body
        food = game_controller.food
        bonus_food = game_controller.bonus_manager.bonus_food or (-1, -1)

        # Example: Flatten coordinates, normalize by grid size if needed
        state = []

        # Flatten snake positions (x,y)
        for segment in snake:
            state.extend(segment)

        # Pad to fixed length if needed (assuming max snake length)
        max_snake_length = 50  # tune this for your game
        if len(snake) < max_snake_length:
            state.extend([-1, -1] * (max_snake_length - len(snake)))

        # Add food and bonus food positions
        state.extend(food)
        state.extend(bonus_food)

        return np.array(state, dtype=np.float32)

    def act(self, game_controller) -> int:
        state = self.preprocess_state(game_controller)
        action = self.agent.act(state, eval_mode=STOP_TRAINING)
        self.last_state = state
        self.last_action = action
        return action

    def remember_and_train(self, game_controller, game_info, done: bool):
        next_state = self.preprocess_state(game_controller)

        # Step 1: Calculate reward
        reward = calculate_reward(self.last_state, self.last_action, next_state, game_info)

        # Step 2: Apply penalty (if any)
        reward = apply_penalty(reward, game_info)

        # Step 3: Optionally update internal state via enhancement
        if "enhancement_type" in game_info:
            self.agent.internal_state = apply_enhancement(
                self.agent.internal_state, game_info["enhancement_type"]
            )

        # Step 4: Store in memory
        if self.last_state is not None and self.last_action is not None:
            self.agent.remember(self.last_state, self.last_action, reward, next_state, done)
        print(f"Reward: {reward:.2f}, Epsilon: {self.agent.epsilon:.3f}, Memory: {len(self.agent.memory)}")

        # Step 5: Train
        self.train()

    def train(self):
        loss = self.agent.replay()
        return loss

    def save(self):
        self.agent.save_model()

    def load(self):
        self.agent.load_model()
