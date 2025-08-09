# ai/dqn_agent.py
import numpy as np
import random
from collections import deque
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from ai.simple_nn import SimpleNN, Activation, Optimizer
from common.utils import clean_input

class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.002,
        gamma: float = 0.99,
        epsilon_start: float = 0.9,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.999,
        memory_size: int = 15000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        model_path: Optional[str] = None,
        seed: Optional[int] = None
    ):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Hyperparameters
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_step_counter = 0

        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Model persistence
        self.model_path = Path(model_path) if model_path else Path(
            os.path.join(os.path.dirname(__file__), "..", "data", "dqn_model")
        )
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize networks
        self.q_network = SimpleNN(
            input_size=state_size,
            hidden_sizes=[128, 128],  # Deeper network
            output_size=action_size,
            learning_rate=learning_rate,
            activation=Activation.RELU,
            output_activation=Activation.LINEAR,
            optimizer=Optimizer.ADAM,
            l2_reg=0.0001,
            clip_value=10.0,
            seed=seed
        )
        
        self.target_network = SimpleNN(
            input_size=state_size,
            hidden_sizes=[128, 128],
            output_size=action_size,
            learning_rate=learning_rate,
            activation=Activation.RELU,
            output_activation=Activation.LINEAR,
            optimizer=Optimizer.ADAM,
            l2_reg=0.0001,
            clip_value=10.0,
            seed=seed
        )
        
        self.update_target_network(full_update=True)
        self.load_model()

    def update_target_network(self, full_update: bool = False) -> None:
        """Update target network either fully or partially (polyak averaging)"""
        if full_update:
            # Full copy for initial sync
            self.target_network.load_state_dict(self.q_network.get_state_dict())
        else:
            # Polyak averaging (tau = 0.001)
            tau = 0.001
            q_state = self.q_network.get_state_dict()
            target_state = self.target_network.get_state_dict()
            
            for i in range(len(target_state['layers'])):
                target_state['layers'][i]['W'] = (
                    tau * q_state['layers'][i]['W'] + 
                    (1 - tau) * target_state['layers'][i]['W']
                )
                target_state['layers'][i]['b'] = (
                    tau * q_state['layers'][i]['b'] + 
                    (1 - tau) * target_state['layers'][i]['b']
                )
            
            self.target_network.load_state_dict(target_state)

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Epsilon-greedy action selection"""
        if not eval_mode and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = clean_input(state)
        q_values = self.q_network.predict(state)  # now explicit
        return int(np.argmax(q_values))


    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store experience in replay buffer with preprocessing"""
        state = clean_input(state)
        next_state = clean_input(next_state)
        self.memory.append((state, action, reward, next_state, done))

    def replay(self) -> Optional[float]:
        """Train on a batch or return diagnostics if not enough memory"""
        if len(self.memory) < self.batch_size:
            # Diagnostic Q average (optional)
            if len(self.memory) >= 8:
                sample = random.sample(self.memory, min(16, len(self.memory)))
                states = np.vstack([clean_input(s[0]) for s in sample])
                q_vals = self.q_network.forward(states, training=False)
                return float(np.mean(q_vals))
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.vstack(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.vstack(next_states)
        dones = np.array(dones)

        current_q = self.q_network.forward(states, training=True)
        next_q = self.target_network.forward(next_states, training=False)
        target_q = current_q.copy()

        best_actions = np.argmax(self.q_network.forward(next_states, training=False), axis=1)
        target_q[np.arange(self.batch_size), actions] = rewards + self.gamma * (
            next_q[np.arange(self.batch_size), best_actions] * (1 - dones)
        )

        loss = self.q_network.train(states, target_q)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.update_target_network()

        return float(loss)

    def save_model(self) -> None:
        """Save model weights and hyperparameters (excluding target_network)"""
        try:
            # Ensure directory exists
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the model architecture and weights
            self.q_network.save(self.model_path.with_suffix('.json'))
            
            # Save additional agent state
            agent_state = {
                'epsilon': self.epsilon,
                'train_step_counter': self.train_step_counter,
                'hyperparameters': {
                    'state_size': self.state_size,
                    'action_size': self.action_size,
                    'gamma': self.gamma,
                    'epsilon_start': self.epsilon,
                    'epsilon_min': self.epsilon_min,
                    'epsilon_decay': self.epsilon_decay,
                    'batch_size': self.batch_size,
                    'target_update_freq': self.target_update_freq
                }
            }
            
            with open(self.model_path.with_suffix('.agent'), 'w') as f:
                json.dump(agent_state, f)
                
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise

    def load_model(self) -> None:
        """Load Q-network weights and hyperparameters (recomputes target_network)"""
        try:
            model_file = self.model_path.with_suffix('.json')
            agent_file = self.model_path.with_suffix('.agent')
            
            if model_file.exists():
                # Load the Q-network
                self.q_network = SimpleNN.load(model_file)
                
                # Load agent state if available
                if agent_file.exists():
                    with open(agent_file, 'r') as f:
                        agent_state = json.load(f)
                    
                    self.epsilon = agent_state.get('epsilon', self.epsilon)
                    self.train_step_counter = agent_state.get('train_step_counter', 0)
                    
                    # Update hyperparameters if they exist
                    if 'hyperparameters' in agent_state:
                        hp = agent_state['hyperparameters']
                        self.gamma = hp.get('gamma', self.gamma)
                        self.epsilon_min = hp.get('epsilon_min', self.epsilon_min)
                        self.epsilon_decay = hp.get('epsilon_decay', self.epsilon_decay)
                        self.batch_size = hp.get('batch_size', self.batch_size)
                        self.target_update_freq = hp.get('target_update_freq', self.target_update_freq)
                
                # Update target network
                self.update_target_network(full_update=True)
                print(f"Model loaded successfully. Epsilon: {self.epsilon:.3f}, Steps: {self.train_step_counter}")
                
        except Exception as e:
            import warnings
            warnings.warn(f"[DQNAgent] Model load failed. Starting fresh. Error: {str(e)}")

    def get_config(self) -> Dict[str, Any]:
        """Return agent configuration for monitoring"""
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'train_steps': self.train_step_counter,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'gamma': self.gamma
        }