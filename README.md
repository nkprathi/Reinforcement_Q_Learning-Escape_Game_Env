Reinforcement_Q_Learning-Escape_Game_Environment

This project implements a **Q-Learning** agent that learns to navigate a grid-based escape game. The agent's goal is to collect coins and reach the goal while avoiding obstacles.


## Core Concepts


### 1. Q-Learning
Q-Learning is a model-free, value-based Reinforcement Learning algorithm. It aims to learn a **policy** that tells an agent what action to take under what circumstances. It does this by learning the **Q-values** (Quality values) for action-state pairs $(s, a)$.

- **State ($s$):** In this environment, the state is represented as `(x, y, coin_mask)`, where $(x, y)$ are coordinates and `coin_mask` is a bitmask representing which coins have been collected.
- **Action ($a$):** The agent can move in 4 directions: `0: Up`, `1: Down`, `2: Right`, `3: Left`.
- **Reward ($r$):** The numerical feedback received from the environment after an action.


---


### 2. The Bellman Equation
The core of Q-Learning is the **Bellman Equation**, which provides a recursive way to update Q-values based on the agent's experience.


The Q-table update rule is defined as:

![bellmaneqn](https://github.com/user-attachments/assets/055c33ec-fad2-4784-8ffc-2862b927d148)


#### Definitions:
- **$Q(s, a)$**: The current value of taking action $a$ in state $s$.
- **$\alpha$ (Alpha - Learning Rate)**: Determines how much of the new information will override the old information ($0 < \alpha \leq 1$).
- **$r$**: The reward received after transitioning from state $s$ to $s'$.
- **$\gamma$ (Gamma - Discount Factor)**: Determines the importance of future rewards ($0 \leq \gamma < 1$). A value close to 1 makes the agent strive for a long-term high reward.
- **$\max_{a'} Q(s', a')$**: The estimate of the optimal future value at the next state $s'$.
- **TD Error**: The segment $[r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$ represents the **Temporal Difference Error**.


---


