# Implementation-of-Iterative-Policy-Evaluation-for-a-Finite-MDP
## Aim

To implement iterative policy evaluation using Gymnasium and estimate the state-value function $V^\pi(s)$ for a fixed random policy.

---
## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

---

## Environment Used

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-based reinforcement learning environment where the agent starts from a start state and tries to reach the goal state without falling into holes.

For the default 4 x 4 FrozenLake map:

| Component | Description |
|---|---|
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching goal, 0 otherwise |
| Terminal states | Goal and holes |

---

## Problem Statement

Evaluate a fixed random policy in the FrozenLake-v1 environment.

The agent follows a random policy, where each of the four actions is selected with equal probability:

$$
\pi(a|s) = \frac{1}{4}
$$

This probability refers to the policy's action-selection probability. The environment transition probabilities are obtained from Gymnasium using `env.P[state][action]`. If `is_slippery=True`, the agent may not move in the intended direction due to stochastic transitions.

The objective is to estimate the state-value function:

$$
V^\pi(s)
$$

---

## Theory

The state-value function under policy $pi$, denoted by $V^\pi(s)$, represents the expected return starting from state $s$ and following policy $pi$.

The Bellman expectation equation is:

```math
V^\pi(s) =
\sum_a \pi(a|s)
\sum_{s'} P(s'|s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
```

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $\pi(a \mid s)$ | Probability of selecting action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $\gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $\pi$ |

---
## Algorithm

1. Create the FrozenLake-v1 environment using Gymnasium.
2. Access the transition model of the environment.
3. Initialize \(V(s)=0\) for all states.
4. Define a random policy where each action has equal probability.
5. For each state:
   - For each action:
     - Read transition probability, next state, reward, and terminal status.
     - Apply the Bellman expectation equation.
6. Repeat until the value function converges.
7. Display the final value function as a 4 x 4 grid.

---

## Program

```python
# Initialize value function
V = np.zeros(n_states)

def policy_evaluation(env, policy, gamma=0.82, theta=1e-8):
    """
    Performs iterative policy evaluation using the Bellman expectation equation.

    Parameters:
        env    : Gymnasium FrozenLake environment
        policy : Fixed policy to be evaluated
        gamma  : Discount factor
        theta  : Convergence threshold

    Returns:
        V         : Estimated state-value function
        iteration : Number of iterations used for convergence
    """
    # Write your code here.
    V = np.zeros(env.observation_space.n)
    iteration = 0
    while True:
        delta = 0
        new_V = np.copy(V)
        for state in range(env.observation_space.n):
            value = 0
            for action in range(env.action_space.n):
                action_prob = policy[state][action]
                for prob, next_state, reward, terminated in env.P[state][action]:
                    value += action_prob * prob * (
                        reward + gamma * V[next_state] * (not terminated)
                    )
            new_V[state] = value
            delta = max(delta, abs(V[state] - new_V[state]))
        V = new_V
        iteration += 1
        if delta < theta:
            break
    return V, iteration

```

---

## Output
<img width="688" height="358" alt="image" src="https://github.com/user-attachments/assets/e24a3a74-dc9b-479e-aace-a393b929a03d" />

---

## Result

Iterative policy evaluation was implemented successfully using the Gymnasium FrozenLake environment. The state-value function for the fixed random policy was estimated using the Bellman expectation equation.

---

## Inference

```text
--> For γ = 0.82, the value function converged after 37 iterations.

--> In the FrozenLake environment, the default setting is is_slippery = True, making the environment stochastic. Therefore, the agent may not always move in the intended direction.

--> A fixed random policy was used, where each action was selected with equal probability (0.25), resulting in relatively low state values because the agent does not always choose the optimal action.

--> States closer to the goal have higher state values, as they have a greater probability of eventually reaching the goal and receiving the reward.

--> Hole states and terminal states have a value of 0 because the episode terminates immediately without any future reward.

--> The discount factor (γ = 0.82) gives moderate importance to future rewards. As a result, the estimated state values are lower than those obtained with larger discount factors such as γ = 0.90 or γ = 0.99.

--> The iterative policy evaluation algorithm successfully estimated the state-value function using the Bellman expectation equation until the value function converged.


```




---


