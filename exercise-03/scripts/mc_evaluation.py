from collections import defaultdict

def mc_evaluation(policy, env, num_episodes, discount_factor=1.0):
  """
  First-visit MC Policy Evaluation. Calculates the value function
  for a given policy using sampling.

  Args:
    policy: A function that maps an observation to action probabilities.
    env: OpenAI gym environment.
    num_episodes: Nubmer of episodes to sample.
    discount_factor: Lambda discount factor.

  Returns:
    A dictionary that maps from state to value.
    The state is a tuple -- containing the players current sum, the dealer's one showing card (1-10 where 1 is ace) 
    and whether or not the player holds a usable ace (0 or 1) -- and the value is a float.
  """

  # Keeps track of sum and count of returns for each state
  # to calculate an average.
  returns_sum = defaultdict(float)
  returns_count = defaultdict(float)

  # The final value function
  V = defaultdict(float)

  # Iterate over the number of episodes.
  for i in range(num_episodes):
      # New game.
      s = env.reset()
      G = 0
      states = []
      # Play one game.
      while(True):
          states.append(s)
          a = policy(s)
          new_s, r, done, none = env.step(a)
          G = G + r
	  # Check if game has finished.
          if done:
              for state in states:
                returns_sum[state] = returns_sum[state] + G
                returns_count[state] = returns_count[state] + 1
              break
          s = new_s
  # For all states that have been visited calculate the v.
  for i in returns_sum:
    V[i] = returns_sum[i]/returns_count[i]
  return V
