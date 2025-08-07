# snake_game/ai/dqn_agent.py
from ai.simple_nn import SimpleNN
import numpy as np
import random
from collections import deque
import json
import os

class DQNAgent:
    def __init__(self, state_size=11, action_size=4, learning_rate=0.002):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        
        # Neural networks
        self.q_network = SimpleNN(state_size, 64, action_size, learning_rate)
        self.target_network = SimpleNN(state_size, 64, action_size, learning_rate)
        self.update_target_network()
        
        self.model_path = os.path.join(os.path.dirname(__file__), "..", "data", "ai_model.json")
        self.load_model()
    
    def update_target_network(self):
        # Copy weights from main to target network
        self.target_network.W1 = self.q_network.W1.copy()
        self.target_network.b1 = self.q_network.b1.copy()
        self.target_network.W2 = self.q_network.W2.copy()
        self.target_network.b2 = self.q_network.b2.copy()
        self.target_network.W3 = self.q_network.W3.copy()
        self.target_network.b3 = self.q_network.b3.copy()
    
    def get_state(self, snake, food, bonus_food, grid_width, grid_height):
        head = snake.get_head()
        head_x, head_y = head
        food_x, food_y = food
        
        # Danger detection
        danger_straight = self._is_collision(snake, head, snake.direction, grid_width, grid_height)
        danger_right = self._is_collision(snake, head, self._turn_right(snake.direction), grid_width, grid_height)
        danger_left = self._is_collision(snake, head, self._turn_left(snake.direction), grid_width, grid_height)
        
        # Direction
        dir_left = snake.direction == (-1, 0)
        dir_right = snake.direction == (1, 0)
        dir_up = snake.direction == (0, -1)
        dir_down = snake.direction == (0, 1)
        
        # Food location relative to head
        food_left = food_x < head_x
        food_right = food_x > head_x
        food_up = food_y < head_y
        food_down = food_y > head_y
        
        state = np.array([
            danger_straight, danger_right, danger_left,
            dir_left, dir_right, dir_up, dir_down,
            food_left, food_right, food_up, food_down
        ], dtype=np.float32)
        
        return state.reshape(1, -1)
    
    def _is_collision(self, snake, head, direction, grid_width, grid_height):
        new_x = head[0] + direction[0]
        new_y = head[1] + direction[1]
        
        # Wall collision
        if new_x < 0 or new_x >= grid_width or new_y < 0 or new_y >= grid_height:
            return 1
        
        # Self collision
        if (new_x, new_y) in snake.body:
            return 1
            
        return 0
    
    def _turn_right(self, direction):
        return {(0, -1): (1, 0), (1, 0): (0, 1), (0, 1): (-1, 0), (-1, 0): (0, -1)}[direction]
    
    def _turn_left(self, direction):
        return {(0, -1): (-1, 0), (-1, 0): (0, 1), (0, 1): (1, 0), (1, 0): (0, -1)}[direction]
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.q_network.forward(state)
        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = np.vstack([e[0] for e in minibatch])
        actions = np.array([e[1] for e in minibatch])
        rewards = np.array([e[2] for e in minibatch])
        next_states = np.vstack([e[3] for e in minibatch])
        dones = np.array([e[4] for e in minibatch])
        
        current_q_values = self.q_network.forward(states)
        next_q_values = self.target_network.forward(next_states)
        
        target = current_q_values.copy()
        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + 0.95 * np.amax(next_q_values[i])
        
        self.q_network.backward(states, target, current_q_values)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.save_model()
    
    def save_model(self):
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            model_data = {
                'W1': self.q_network.W1.tolist(),
                'b1': self.q_network.b1.tolist(),
                'W2': self.q_network.W2.tolist(),
                'b2': self.q_network.b2.tolist(),
                'W3': self.q_network.W3.tolist(),
                'b3': self.q_network.b3.tolist(),
                'epsilon': self.epsilon
            }
            with open(self.model_path, 'w') as f:
                json.dump(model_data, f)
        except Exception as e:
            print(f"Error saving AI model: {e}")
    
    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'r') as f:
                data = json.load(f)
                self.q_network.W1 = np.array(data['W1'])
                self.q_network.b1 = np.array(data['b1'])
                self.q_network.W2 = np.array(data['W2'])
                self.q_network.b2 = np.array(data['b2'])
                self.q_network.W3 = np.array(data['W3'])
                self.q_network.b3 = np.array(data['b3'])
                # Also copy to target network
                self.update_target_network()
        else:
            print("Model file not found, starting fresh.")