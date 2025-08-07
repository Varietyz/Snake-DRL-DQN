# 🐍 Snake DRL/DQN

**Purpose**
- A simple GUI Snake game built with a modular, SoC-compliant architecture using Python and Tkinter
- Designed for learning and extendability, the game cleanly separates logic, rendering, and control layers
- Includes both human-playable and AI-controlled modes using Deep Reinforcement Learning (DQN)

**Features**
- Classic Snake gameplay:
  - Move the snake with arrow keys
  - Eat red food to grow longer
  - Die on collision with wall or self
- AI Mode:
  - Deep Q-Network (DQN) implementation
  - Self-learning capability
  - Model saving/loading
- Architecture:
  - Auto-restartable codebase with clear logic boundaries
  - Central config for easy tuning (speed, size, colors)
  - Clean modular structure for easy testing and extension
- Extras:
  - Bonus food items
  - Score tracking
  - Visual feedback

**Controls**
- Human Mode:
  - Arrow keys – Move the snake
  - Game Over – When hitting wall or self
- AI Mode:
  - Automatic self-play
  - Press 'Space' to toggle AI on/off
  - Model saves automatically

**AI Implementation**
- Deep Q-Learning (DQN) with:
  - Experience replay buffer
  - Target network
  - ε-greedy exploration
  - Custom reward shaping
- State representation:
  - Danger detection (straight/left/right)
  - Current direction
  - Relative food position
- 3-layer neural network (64 hidden units)

## Files
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