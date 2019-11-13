
import math

import numpy as np


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment. env.P represents the transition probabilities
        of the environment.
        theta: Stopping threshold. If the value of all states changes less
        than theta
        in one iteration we are done.
        discount_factor: lambda time discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value
        function.
    """

    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])

    while True:
        # Copy the old value function for convergence check.
        V_old = np.copy(V)
        for s in range(env.nS):
            best_action = 0
            best_value = -math.inf
            # Find the action with the highest expected reward.
            for a in range(env.nA):
                action_reward = env.P[s][a][0][2]
                expected_reward = V[env.P[s][a][0][1]]
                v_tmp = action_reward + discount_factor * expected_reward
                if(v_tmp > best_value):
                    best_value = v_tmp
                    best_action = a
            # Assign the new values to policy and value function.
            policy[s][:] = 0
            policy[s][best_action] = 1
            V[s] = best_value
        if abs(np.sum(V_old) - np.sum(V)) < theta:
            break
    return policy, V
