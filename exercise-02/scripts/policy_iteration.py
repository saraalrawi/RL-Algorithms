
import math

import numpy as np


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the
    environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of
        the environment.
        env.P[s][a] is a list of transition tuples
        (prob, next_state,reward, done).
        theta: We stop evaluation once our value function change is less
        than theta for all states.
        discount_factor: gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        # Copy the old value function for convergence check.
        V_old = np.copy(V)
        for s in range(env.nS):
            v_tmp = 0
            for a in range(env.nA):
                action_reward = env.P[s][a][0][2]
                expected_reward = V[env.P[s][a][0][1]]
                # Add the expected value for a given action and state times the
                # probability for this action.
                v_tmp += policy[s][a] * (action_reward +
                                         discount_factor * expected_reward)
            V[s] = v_tmp
        # Check for convergence criteria.
        if abs(np.sum(V_old) - np.sum(V)) < theta:
            break
    return np.array(V)


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
        policy, env, discount_factor.
        discount_factor: Lambda discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each
        state s contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    V = np.zeros(env.nS)
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        V = policy_eval(policy, env, discount_factor)
        # Copy array to check for convergence.
        policy_old = np.copy(policy)
        for s in range(env.nS):
            # Find out the best action for the current state.
            best_action = 0
            best_value = -math.inf
            for a in range(env.nA):
                if(V[env.P[s][a][0][1]] > best_value):
                    best_value = V[env.P[s][a][0][1]]
                    best_action = a
            # Set the action deterministic.
            policy[s][:] = 0
            policy[s][best_action] = 1
        # Check if further loop is necessary.
        if np.array_equal(policy, policy_old):
            break
    return policy, V
