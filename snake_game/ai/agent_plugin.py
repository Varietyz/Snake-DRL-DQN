# ai/agent_plugin.py
import numpy as np
from ai.dqn_agent import DQNAgent
from ai.game.enhancements import apply_enhancement
from ai.game.rewards import calculate_reward
from ai.game.penalties import apply_penalty
from config import STOP_TRAINING, GRID_WIDTH, GRID_HEIGHT

class AgentPlugin:
    def __init__(self, state_size: int, action_size: int):
        self.agent = DQNAgent(state_size=state_size, action_size=action_size)
        self.last_state = None
        self.last_action = None

    def preprocess_state(self, game_controller) -> np.ndarray:
        snake = game_controller.snake.body
        head = snake[-1]
        food = game_controller.food
        bonus_food = game_controller.bonus_manager.bonus_food or (-1, -1)
        grid_width = GRID_WIDTH
        grid_height = GRID_HEIGHT

        state = []

        # 1. Snake body positions (50 segments max * 2 coordinates)
        for segment in snake:
            state.extend(segment)
        max_snake_length = 50
        if len(snake) < max_snake_length:
            state.extend([-1, -1] * (max_snake_length - len(snake)))
        # Total so far: 50 * 2 = 100 elements

        # 2. Head position relative to walls (4 elements)
        state.extend([
            head[0] / grid_width,          # normalized x position
            head[1] / grid_height,         # normalized y position
            (grid_width - head[0]) / grid_width,    # distance to right wall
            (grid_height - head[1]) / grid_height   # distance to bottom wall
        ])
        # Total: 100 + 4 = 104 elements

        # 3. Direction information (4 elements one-hot encoded)
        direction = game_controller.snake.direction
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # right, left, down, up
        state.extend([1 if direction == d else 0 for d in directions])
        # Total: 104 + 4 = 108 elements

        # 4. Food and bonus positions relative to head (4 elements)
        state.extend([
            (food[0] - head[0]) / grid_width,      # relative x position
            (food[1] - head[1]) / grid_height      # relative y position
        ])
        if bonus_food != (-1, -1):
            state.extend([
                (bonus_food[0] - head[0]) / grid_width,
                (bonus_food[1] - head[1]) / grid_height
            ])
        else:
            state.extend([0, 0])  # no bonus food present
        # Total: 108 + 4 = 112 elements

        # 5. Danger detection (4 elements)
        danger_states = []
        for dx, dy in directions:
            danger = 0
            x, y = head[0] + dx, head[1] + dy
            
            # Check for wall collision
            if x < 0 or x >= grid_width or y < 0 or y >= grid_height:
                danger = 1
            # Check for body collision (excluding tail unless growing)
            elif (x, y) in list(snake)[:-1]:
                danger = 1
            danger_states.append(danger)
        state.extend(danger_states)
        # Final total: 112 + 4 = 116 elements

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
