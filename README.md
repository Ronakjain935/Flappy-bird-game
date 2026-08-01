# 🐤 Flappy Bird Deep Q-Learning (DQN) AI Agent

An autonomous Reinforcement Learning (RL) agent trained to play **Flappy Bird** using **Deep Q-Learning (DQN)** in PyTorch and OpenAI Gymnasium.

---

## 📌 Project Overview

This project implements a Deep Q-Network (DQN) agent that learns to play the Flappy Bird game from scratch. The agent interacts with the `flappy-bird-gymnasium` environment, receiving numerical state vector observations (such as horizontal/vertical distances to next pipes, bird velocity, etc.) and making real-time flapping decisions to maximize its score.

### 🌟 Key Features
- **Deep Q-Network (DQN)**: Multi-Layer Perceptron (MLP) architecture built with PyTorch.
- **Experience Replay Memory**: Circular replay buffer (`deque`) to store transitions and break correlation between consecutive samples.
- **Target Network Synchronization**: Double-network setup (Policy Network + Target Network) for stable Q-learning target values.
- **Epsilon-Greedy Exploration Strategy**: Dynamic $\epsilon$-decay strategy balancing exploration of new states with exploitation of learned Q-values.
- **Configurable Hyperparameters**: Easily customizable RL parameters stored cleanly in `parameters.yaml`.
- **Interactive Human Play Mode**: Included Pygame script to play the game manually using the spacebar.

---

## 📂 Project Structure

```
flappy-bird-game/
├── agent.py               # Main RL Agent, training loop, optimization logic & CLI
├── dqn.py                 # PyTorch Neural Network architecture (DQN)
├── experience_replay.py   # ReplayMemory buffer for experience sampling
├── flappy_bird.py         # Human play script using Pygame controls
├── parameters.yaml        # Training hyperparameters configuration
└── README.md              # Project documentation
```

---

## ⚙️ Hyperparameters (`parameters.yaml`)

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `alpha` (Learning Rate) | `0.001` | Learning rate for the Adam optimizer |
| `gamma` (Discount Factor) | `0.99` | Discount factor for future rewards |
| `epsilon_init` | `1.0` | Initial exploration probability (100% random actions) |
| `epsilon_min` | `0.05` | Minimum exploration probability bound |
| `epsilon_decay` | `0.9995` | Multiplicative decay factor per episode |
| `replay_memory_size` | `100,000` | Maximum capacity of the replay buffer |
| `mini_batch_size` | `32` | Batch size sampled from replay memory for training |
| `network_sync_rate` | `10` | Frequency (steps) of copying policy weights to target network |
| `reward_threshold` | `1000` | Episode reward threshold to trigger next episode |

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Ensure you have **Python 3.8+** installed. Install the required dependencies:

```bash
pip install --no-deps -r requirements.txt
```



---

## 🎮 How to Run

### 1️⃣ Train the AI Agent
To start training a new DQN agent from scratch:

```bash
python agent.py --train
```
- Training logs and the best performing model (`flappybird.pt`) will be saved in the `runs/` directory.

### 2️⃣ Test / Watch the Trained Agent
To watch the trained agent play the game live with visual rendering:

```bash
python agent.py
```

### 3️⃣ Play Manually (Human Mode)
To play Flappy Bird manually using your keyboard:

```bash
python flappy_bird.py
```
### 4️⃣ Launch the Streamlit Demo Web App 🌐
To launch the interactive demo web app in your browser:

```bash
streamlit run app.py
```
- Includes live AI gameplay, real-time Q-value telemetries, hyperparameter tuning lab, Plotly analytics, and interactive model state simulator.

---


## 🧠 Model Architecture & Deep Q-Learning

```
State Input (12 features) ──► Linear(256) ──► ReLU ──► Linear(2) ──► Q-Values (No Flap / Flap)
```

1. **State Space**: 12 continuous values representing game state dynamics (e.g., bird $y$-position, velocity, horizontal/vertical distances to upcoming pipe gaps).
2. **Action Space**: Discrete(2)
   - `0`: Do nothing
   - `1`: Flap wings
3. **Loss Function**: Mean Squared Error (MSE) loss calculated between predicted Q-values $Q(s, a)$ and Target Q-values $r + \gamma \max_{a'} Q_{target}(s', a')$.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
