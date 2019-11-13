
from collections import defaultdict

import numpy as np


def create_random_policy(nA):
    """
  Creates a random policy function.

  Args:
    nA: Number of actions in the environment.

  Returns:
    A function that takes an observation as input and returns a vector
    of action probabilities
  """
    A = np.ones(nA, dtype=float) / nA

    def policy_fn(observation):
        return A

    return policy_fn


def create_greedy_policy(Q):
    """
  Creates a greedy policy based on Q values.

  Args:
    Q: A dictionary that maps from state -> action values

  Returns:
    A function that takes an observation as input and returns a vector
    of action probabilities.
  """
    def policy_fn(state):
        pol = [0.0, 0.0]
        pol[np.argmax(Q[state])] = 1.0
        return pol 
    return policy_fn


def mc_control_importance_sampling(env, num_episodes, behavior_policy,
                                   discount_factor=1.0):
    """
  Monte Carlo Control Off-Policy Control using Importance Sampling.
  Finds an optimal greedy policy.

  Args:
    env: OpenAI gym environment.
    num_episodes: Nubmer of episodes to sample.
    behavior_policy: The behavior to follow while generating episodes.
        A function that given an observation returns a vector of probabilities
        for each action.
    discount_factor: Lambda discount factor.

  Returns:
    A tuple (Q, policy).
    Q is a dictionary mapping state -> action values.
    policy is a function that takes an observation as an argument and returns
    action probabilities. This is the optimal greedy policy.
  """

    # The final action-value function.
    # A dictionary that maps state -> action values
    # Keeps track of sum and count of returns for each state
    # to calculate an average.

    # Count the occurences of each [state, action] pair.
    returns_count = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # Our greedily policy we want to learn
    target_policy = create_greedy_policy(Q)
    for i in range(num_episodes):
        # episode is an array of arrays[state, action]
        episode = []  
        state = env.reset()
        G = 0.0
        importance = []
        while True:
            # determine probability from behavioral policy
            state_probability = behavior_policy(state)
            action = np.random.choice(2, p=state_probability)
            next_state, reward, done, none = env.step(action)
            episode.append([state, action])
            # calculate sampling term.
            importance_update = (target_policy(state)[action] / 
                                 behavior_policy(state)[action])
            # store sampling term.
            importance.append(importance_update)
            G += reward
            if done:
                # we iterate backwards over all states in the episode.
                for j in range(len(episode)):
                    # Get values.
                    state = episode[j][0]
                    action = episode[j][1]
                    returns_count[state][action] += 1.0
                    # Calculate future importance.
                    G_sampled = G * np.prod(importance[j:])
                    # Running mean.
                    alpha = 1 / returns_count[state][action]
                    # Calculate new Q with running mean.
                    Q[state][action] += alpha * (G_sampled - Q[state][action])
                    # probability will be zero. In this case break.
                    if action != np.argmax(target_policy(state)):
                        break
                break
            state = next_state
    return Q, target_policy
