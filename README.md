# Noisy DQN with Experience Replay: Atari Breakout

A **Noisy DQN** implementation with **Experience Replay** applied to `BreakoutNoFrameskip-v4` from OpenAI Gymnasium. This project replaces the traditional epsilon-greedy exploration strategy with **NoisyLinear layers** (Fortunato et al., 2018) that inject learned parametric noise directly into the network weights. Combined with a **Replay Buffer** to break temporal correlations and a **Target Network** to stabilize Q-value targets.

## Project Overview

The agent learns to play Atari Breakout using a convolutional neural network with noisy fully connected layers to approximate Q-values. The implementation incorporates three key components:

1. **NoisyLinear Layers** — The standard fully connected layers are replaced with NoisyLinear layers that add factorised Gaussian noise to their weights and biases. The noise magnitude (sigma) is a trainable parameter — the network **learns how much to explore** in each part of the weight space. As training progresses, sigma shrinks in well-learned regions and the agent naturally transitions from exploration to exploitation without any epsilon schedule.
2. **Experience Replay Buffer** — A memory-efficient buffer that stores individual frames (uint8) and reconstructs stacked states on demand, breaking the correlation between consecutive samples while minimizing memory usage.
3. **Target Network** — A periodically-updated frozen copy of the policy network provides stable TD targets, solving the moving target problem.

### Why Noisy Nets Over Epsilon-Greedy?

Epsilon-greedy exploration is **state-independent** — it explores with the same probability regardless of what the agent sees. Noisy DQN provides **state-dependent exploration**: the noise perturbs the Q-value estimates differently depending on the input state, so the agent can explore more in unfamiliar states and exploit in well-understood ones. This is achieved without any hyperparameter tuning of exploration schedules.

### Core Features

- **Modular Architecture:** Separated into distinct files for the environment, agent, network, replay buffer, utilities, and training logic.
- **No Epsilon Schedule:** Exploration is entirely driven by learnable noise parameters — no epsilon decay to tune.
- **Step-Based Training Loop:** Training runs for a fixed number of agent steps (not episodes), matching the original DeepMind approach. All logging and checkpointing are step-based.
- **YAML Configuration:** All hyperparameters, paths, and execution modes are controlled via `config.yaml` — no need to edit code between experiments.
- **GPU/MPS Support:** Automatic device detection for CUDA, Apple Silicon (MPS), or CPU fallback.
- **Persistent Storage:**
  - `.pth` files store the trained model weights.
  - `.npz` files store the full training history (rewards, losses, step count).
- **Resume Training:** Interrupt training at any time (Ctrl+C) and resume from the last checkpoint without losing progress.
- **Visualization:** Automatic generation of reward and loss training curves at each checkpoint, plus a deployment mode with action distribution logging.

---

## Project Structure

```
Noisy-DQN-ER/
├── config.yaml            # All hyperparameters and settings
├── noisy_linear.py        # NoisyLinear layer with factorised Gaussian noise
├── q_network.py           # CNN Q-Network with NoisyLinear fully connected layers
├── replay_buffer.py       # Memory-efficient Experience Replay Buffer
├── dqn_agent.py           # Noisy DQN agent (policy + target networks)
├── environment.py         # Gym environment setup with Atari preprocessing
├── utils.py               # Config loading, plotting, and deployment visualization
├── training_script.py     # Step-based training loop and main entry point
└── noisy_dqn_results/     # Generated outputs (model, history, plots)
```

---

## Key Components

| Component | Purpose |
|---|---|
| **NoisyLinear Layers** | Replace standard Linear layers with noise-injected versions; sigma is learned via gradient descent, providing automatic state-dependent exploration |
| **Factorised Gaussian Noise** | Uses O(p+q) noise vectors instead of O(p·q) independent noise, transformed by f(x) = sign(x) · √\|x\| for heavier tails |
| **Experience Replay Buffer** | Stores individual frames (uint8) and reconstructs stacked states on demand, breaking temporal correlation while minimizing memory usage |
| **Target Network** | A frozen copy of the policy network, updated every 10,000 steps, provides stable TD targets |
| **Double DQN Target Calculation** | Policy network selects the best action; target network evaluates it — reduces overestimation bias |
| **Dedicated Warmup** | `train()` calls `warmup()` internally to fill the replay buffer with random-action transitions before the training loop begins, avoiding per-step buffer-size checks during learning |
| **Periodic Checkpointing** | Saves model, history, and training plots every 100,000 steps to protect against crashes |

### NoisyLinear Mechanism

```
Standard Linear:    y = Wx + b
NoisyLinear:        y = (μʷ + σʷ · εʷ) · x + (μᵇ + σᵇ · εᵇ)

where:
  μʷ, σʷ, μᵇ, σᵇ  — learnable parameters (optimized by gradient descent)
  εʷ, εᵇ           — factorised noise (resampled each learning step)
```

As training progresses, gradient descent drives sigma toward zero in well-learned regions, naturally reducing exploration where the agent is already confident.

### Double DQN Target Calculation

```
Standard DQN:   target = rₜ + γ · maxₐ' Q_target(sₜ₊₁, a')
Double DQN:     target = rₜ + γ · Q_target(sₜ₊₁, argmaxₐ' Q_policy(sₜ₊₁, a'))
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch
- Gymnasium with Atari support (`ale-py`)
- NumPy, Matplotlib, PyYAML

### Installation

```bash
pip install torch gymnasium ale-py numpy matplotlib pyyaml
```

### Running the Script

Control the execution mode via `config.yaml`:

```yaml
training:
  mode: "new"      # Train from scratch
  # mode: "resume" # Load saved model and continue training
  # mode: "deploy" # Watch the trained agent play
```

Then run:

```bash
python training_script.py
```

---

## Configuration

All parameters are managed in [`config.yaml`](config.yaml). The config is organized into logical sections:

### Training Loop

| Parameter | Default | Description |
|---|---|---|
| `num_steps` | 2,500,000 | Total agent steps to train for (each step = 4 frames) |
| `target_reward` | 40 | Average reward for early stopping (per-life, not per-game) |
| `print_every` | 10,000 | Steps between console progress logs |
| `checkpoint_every` | 100,000 | Steps between model/plot checkpoints (0 to disable) |
| `plot_window` | 500 | Moving average window for reward curves and early stopping |

### Agent Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `optimizer` | adam | Optimizer type (`adam` or `rmsprop`) |
| `learning_rate` | 0.00025 | Optimizer learning rate |
| `gamma` | 0.99 | Discount factor for future rewards |
| `loss_function` | huber | Loss type (`huber` or `mse`) |
| `grad_clip_norm` | 1.0 | Maximum gradient norm for clipping |
| `clip_rewards` | true | Clip rewards to [-1, 1] for stable gradients |
| `sigma_init` | 0.5 | Initial noise scale for NoisyLinear layers |

### Experience Replay Buffer

| Parameter | Default | Description |
|---|---|---|
| `capacity` | 1,000,000 | Maximum transitions stored (FIFO eviction) |
| `batch_size` | 32 | Mini-batch size for training |
| `learning_starts` | 50,000 | Warmup transitions before learning begins |

### Target Network

| Parameter | Default | Description |
|---|---|---|
| `update_freq` | 10,000 | Steps between target network weight copies |

### Other

| Parameter | Default | Description |
|---|---|---|
| `seed` | 42 | Random seed for reproducibility |

---

## References

- Fortunato, M. et al. (2018). *Noisy Networks for Exploration.* ICLR.
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518(7540), 529-533.
- van Hasselt, H. et al. (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI.
