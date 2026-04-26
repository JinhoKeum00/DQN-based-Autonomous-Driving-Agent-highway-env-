# DQN-based Autonomous Driving Agent

Reinforcement Learning project for autonomous driving using `highway-env`.  

---

## Overview

In a standard DQN setup, the agent tends to avoid collisions by driving slowly in a safe lane, failing to perform overtaking or efficient driving.  
This behavior is known as the **Lazy Agent problem (local optima)**.

### Goal
- Encourage **active lane changing and high-speed driving**
- Improve policy **without modifying the reward function**
- Solve the problem through **algorithmic improvements only**

---

## Method

We enhanced the baseline DQN by improving perception, decision-making, exploration, and learning.

### 1. Perception – Ego Attention (Social Attention)
- Models interactions between ego vehicle and surrounding vehicles
- Uses attention mechanism to focus on important entities

### 2. Decision – Dueling Network
- Separates state value and advantage
- Improves learning efficiency

### 3. Exploration – NoisyNet
- Replaces epsilon-greedy with parameter noise
- Enables state-dependent exploration

### 4. Learning – Double DQN + N-step Learning
- Reduces Q-value overestimation (Double DQN)
- Handles delayed rewards (N-step, N=5)

---

## Environments

Experiments were conducted on multiple driving scenarios:

- **Highway-v0**: Dense traffic, high interaction, unstable learning
- **Merge-v0**: Simpler structure, stable convergence
- **Roundabout-v0**: Complex interactions, high performance but unstable
- **Intersection-v0**: Sparse rewards, difficult exploration

---

## Results & Analysis

### Key Findings

- The agent achieved **stable driving and longer survival** in highway scenarios
- Merge environment showed **stable convergence**
- Roundabout achieved **high scores (~9.7)** but had instability
- Intersection showed **partial success with frequent collapse**

### Observed Issues

- **Q-value divergence**
- **Gradient explosion**
- **Training instability despite reward increase**

This indicates that reward improvement alone can be misleading without stable value estimation. :contentReference[oaicite:1]{index=1}

---

## Key Insights

- Reward shaping is not necessary if the model architecture is well designed
- Exploration strategy (NoisyNet) strongly affects training stability
- Complex environments require better stability mechanisms

---

## Tech Stack

- Python
- PyTorch
- highway-env
- Reinforcement Learning (DQN variants)

---

## Future Work

- Stabilize training with:
  - Distributional DQN (C51, QR-DQN)
  - Better target updates (soft update)
- Improve exploration stability (fix NoisyNet sigma issue)
- Apply to more complex real-world scenarios

---

## References

- Dueling Network: https://arxiv.org/abs/1511.06581
- NoisyNet: https://arxiv.org/abs/1706.10295
- Social Attention: https://arxiv.org/pdf/1911.12250
- highway-env: https://highway-env.farama.org/
