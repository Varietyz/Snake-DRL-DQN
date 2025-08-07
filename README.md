# 🐍 AI Snake Game

A modern Snake game implementation featuring both human and AI players, built with Python and Tkinter using Deep Q-Learning.

## ✨ Features

### 🎮 Gameplay
- Classic Snake mechanics with smooth controls
- Progressive difficulty (speed increases with score)
- Bonus food system with time-limited yellow food (+5 points)
- Visual feedback when bonus food is about to disappear

### 🤖 AI Player
- Deep Q-Network (DQN) reinforcement learning agent
- Self-learning capability with experience replay
- Neural network implemented from scratch (no TensorFlow/PyTorch)
- Persistent model that improves over time
- Toggle between AI and human control with spacebar

### 🛠 Technical
- Clean architecture with separation of concerns:
  - Core game logic (`core/`)
  - AI implementation (`ai/`)
  - UI components (`ui/`)
- Custom neural network implementation
- Configurable parameters (speed, grid size, colors, AI settings)
- Score persistence with JSON storage

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Tkinter (usually included with Python)
- NumPy (`pip install numpy`)

### Installation
```bash
git clone https://github.com/yourusername/snake-game.git
cd snake-game
pip install -r requirements.txt
```

### Running the Game
```bash
python -m snake_game.main
```

## 🕹 Controls
| Key | Action |
|-----|--------|
| ↑ ↓ ← → | Move snake (human mode) |
| Space | Toggle AI/human mode |
| ESC | Quit game |

## ⚙ Configuration
Customize game parameters in `config.py`:
```python
GRID_SIZE = 20          # Pixel size of each grid cell
GRID_WIDTH = 30         # Grid width in cells
GRID_HEIGHT = 20        # Grid height in cells
SPEED = 150             # Initial speed (ms per frame)
AI_SPEED = 30           # AI speed (much faster)
BONUS_FOOD_CHANCE = 0.15 # 15% chance for bonus food
```

## 🤖 About the AI
The DQN agent uses:
- 3-layer neural network (input, hidden, output)
- 11-dimensional state space (danger detection + food location)
- Experience replay with 100,000 memory capacity
- Epsilon-greedy exploration (decays from 1.0 to 0.01)
- Model auto-saves to `data/ai_model.json`

## 📂 Project Structure
```
├─ 📘 README.md
├─ 📄 requirements.txt
└─ 📂 snake_game
    ├─ 📂 ai
    │   ├─ 🐍 dqn_agent.py
    │   ├─ 🐍 simple_nn.py
    │   └─ 🐍 __init__.py
    ├─ 🐍 config.py
    ├─ 📂 core
    │   ├─ 🐍 food.py
    │   ├─ 🐍 score.py
    │   └─ 🐍 snake.py
    ├─ 📂 data
    │   ├─ 🔧 ai_model.json
    │   └─ 🔧 score.json
    ├─ 🐍 main.py
    └─ 📂 ui
        ├─ 🐍 ai_controller.py
        ├─ 🐍 bonus_manager.py
        ├─ 🐍 controller.py
        ├─ 🐍 input_handler.py
        └─ 🐍 renderer.py
```