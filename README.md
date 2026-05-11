# EX.1 Pole Balancing using Reinforcement Learning
## Date: 09-05-2026

## Aim
The aim of this project is to develop a Reinforcement Learning agent that learns to balance a pole on a moving cart by interacting with the environment and maximizing cumulative rewards.

# Algorithm

## Q-Learning Algorithm for Pole Balancing

### Step 1
Initialize the CartPole environment.

### Step 2
Initialize the Q-table with zeros.

### Step 3
Observe the current state of the environment.

### Step 4
Choose an action using the epsilon-greedy policy:
- Exploration → Random action
- Exploitation → Best action from Q-table

### Step 5
Perform the selected action.

### Step 6
Receive:
- Reward
- Next state
- Done status

### Step 7
Update the Q-value using:

Q(s,a) = Q(s,a) + α [ r + γ max Q(s',a') − Q(s,a) ]

Where:
- α → Learning rate
- γ → Discount factor

### Step 8
Repeat the process until the episode ends.

### Step 9
Train the agent for multiple episodes to improve balancing performance.

---

# Program

```
#Pole Balancing

import numpy as np
import gymnasium as gym

# Create environment (modern version)
env = gym.make("CartPole-v1")

# Discretization
buckets = (1, 1, 6, 12)
Q = np.zeros(buckets + (env.action_space.n,))

# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 3000

# Discretize continuous state
def discretize(obs):
    upper = [2.4, 3.0, 0.2095, 3.5]
    lower = [-2.4, -3.0, -0.2095, -3.5]

    ratios = [(obs[i] - lower[i]) / (upper[i] - lower[i]) for i in range(len(obs))]
    new_obs = [int(r * (buckets[i] - 1)) for i, r in enumerate(ratios)]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]

    return tuple(new_obs)

# --- TRAINING ---
for ep in range(episodes):
    obs, _ = env.reset()
    state = discretize(obs)
    done = False

    while not done:
        # ε-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Slightly improved reward (optional but simple)
        angle = next_obs[2]
        reward = 1 - abs(angle)

        next_state = discretize(next_obs)

        # Q-learning update
        Q[state][action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state][action]
        )

        state = next_state

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if ep % 500 == 0:
        print(f"Episode {ep}, Epsilon: {epsilon:.3f}")

print("Training Completed!")


# --- TESTING ---
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    state = discretize(obs)
    action = np.argmax(Q[state])

    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    total_reward += reward

env.close()
print("Test Reward:", total_reward)

```

# Output

<img width="340" height="168" alt="Screenshot 2026-05-09 141059" src="https://github.com/user-attachments/assets/c1f75fee-50ac-47fd-97bb-d67af1b2392b" />




**# Result**

The Reinforcement Learning agent successfully learned to balance the pole using the Q-Learning algorithm.  
The reward increased gradually with training episodes, demonstrating that the agent improved its balancing strategy through learning and interaction with the environment.


