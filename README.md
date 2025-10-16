# Atari_RL

## Overview

This repository contains PyTorch implementations of Deep Reinforcement Learning algorithms for playing Atari games. The project leverages the Gymnasium library (the successor to OpenAI Gym) to provide a standardized interface for training agents on classic Atari environments. Currently, the repository includes two state-of-the-art RL agents:

- **PPO (Proximal Policy Optimization)** trained on Pong
- **Double DQN (Deep Q-Network)** trained on Breakout

These implementations serve as educational examples and baselines for reinforcement learning research in Atari environments.

## Agents

### 1. PPO for Pong

The PPO agent uses the Proximal Policy Optimization algorithm, a policy gradient method that balances exploration and exploitation through a clipped objective function. This implementation is optimized for the Pong-v5 environment from Gymnasium.

**Key Features:**
- Policy and value networks with shared features
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective for stable training
- Parallel environment execution

### 2. Double DQN for Breakout

The Double DQN agent implements an improved version of the classic Deep Q-Network algorithm, addressing overestimation bias by decoupling action selection from action evaluation. This agent is trained on the Breakout-v5 environment.

**Key Features:**
- Experience replay buffer for sample efficiency
- Target network for stable learning
- Double Q-learning to reduce overestimation
- Frame stacking and preprocessing

## Repository Structure

The repository is organized into separate folders for each agent implementation:

```
atari_rl/
├── ppo_pong/
│   ├── train.py           # Training script for PPO agent
│   ├── model.py           # PPO neural network architecture
│   ├── agent.py           # PPO agent implementation
│   └── utils.py           # Utility functions and helpers
├── dqn_breakout/
│   ├── train.py           # Training script for DQN agent
│   ├── model.py           # DQN neural network architecture
│   ├── agent.py           # Double DQN agent implementation
│   ├── replay_buffer.py   # Experience replay buffer
│   └── utils.py           # Utility functions and helpers
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── LICENSE               # License information
```

Each agent folder contains:
- `train.py`: Main training script with hyperparameters and training loop
- `model.py`: Neural network architecture definition
- `agent.py`: Agent logic including action selection and learning updates
- `utils.py`: Helper functions for preprocessing, logging, and visualization

## Installation

Follow these steps to set up the environment and install dependencies:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Hasakij/Atari_RL.git
   cd Atari_RL
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   
   On Linux/Mac:
   ```bash
   source venv/bin/activate
   ```
   
   On Windows:
   ```bash
   venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

**Note:** Training Atari agents requires a machine with a CUDA-compatible GPU for reasonable training times. CPU-only training is possible but will be significantly slower.

## Training

Once the environment is set up, you can train each agent using the following commands:

### Train PPO on Pong

```bash
python ppo_pong/train.py
```

**Optional arguments:**
- `--total-timesteps`: Total number of timesteps to train (default: 10000000)
- `--learning-rate`: Learning rate for the optimizer (default: 2.5e-4)
- `--num-envs`: Number of parallel environments (default: 8)
- `--wandb`: Enable Weights & Biases logging

### Train Double DQN on Breakout

```bash
python dqn_breakout/train.py
```

**Optional arguments:**
- `--total-timesteps`: Total number of timesteps to train (default: 10000000)
- `--learning-rate`: Learning rate for the optimizer (default: 1e-4)
- `--buffer-size`: Size of replay buffer (default: 100000)
- `--wandb`: Enable Weights & Biases logging

## Results

Training progress and results are logged using Weights & Biases (wandb). You can track metrics such as:
- Episode rewards
- Training loss
- Q-values (for DQN)
- Policy entropy (for PPO)

### PPO on Pong

**Weights & Biases Run:** [Link to be added]

**Training Performance:**
- Final Average Reward: TBD
- Training Time: TBD
- Number of Episodes: TBD

![PPO Pong Gameplay GIF](placeholder_ppo_pong.gif)

### Double DQN on Breakout

**Weights & Biases Run:** [Link to be added]

**Training Performance:**
- Final Average Reward: TBD
- Training Time: TBD
- Number of Episodes: TBD

![DQN Breakout Gameplay GIF](placeholder_dqn_breakout.gif)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Gymnasium](https://gymnasium.farama.org/) for providing the Atari environments
- Original papers: [PPO](https://arxiv.org/abs/1707.06347) and [Double DQN](https://arxiv.org/abs/1509.06461)
- [CleanRL](https://github.com/vwxyzjn/cleanrl) for implementation inspiration
