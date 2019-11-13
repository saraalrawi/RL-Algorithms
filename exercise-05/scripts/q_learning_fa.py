import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import random
from collections import namedtuple

if "../../" not in sys.path:
  sys.path.append("../../")
#from lib.envs.mountain_car import MountainCarEnv
from mountain_car import MountainCarEnv

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

class NeuralNetwork():
  """
  Neural Network class based on TensorFlow.
  """

  def __init__(self):
    self.y_est = None
    self.y = tf.placeholder(shape=(1, 3), dtype=tf.float64, name='y')
    self.X = tf.placeholder(shape=(1, 2), dtype=tf.float64, name='X')
    self.train = None
    self.loss = None
    self._build_model()

  def _build_model(self):

    """
    Creates a neural network, e.g. with two
    hidden fully connected layers and 20 neurons each). The output layer
    has #A neurons, where #A is the number of actions and has linear activation.
    Also creates its loss (mean squared loss) and its optimizer (e.g. Adam with
    a learning rate of 0.0005). For initialization, you can simply use a uniform
    distribution (-0.5, 0.5), or something different.
    """
    dense1 = tf.layers.dense(inputs=self.X, units=20, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1, units=20, activation=tf.nn.relu)
    # output layer 3 actions and linear activation
    self.y_est = tf.layers.dense(inputs=dense2, units=3, activation=None)

    # squared delta output and test data y
    deltas = tf.square(self.y_est - self.y)
    self.loss = tf.reduce_sum(deltas)
    # adam optimizer with learning rate of 0.05%
    optimizer = tf.train.AdamOptimizer(0.0005)
    # tensorflow uses the model to minimize the loss
    self.train = optimizer.minimize(self.loss)
  
  def predict(self, sess, states):
    """
    Args:
      sess: TensorFlow session
      states: array of states for which we want to predict the actions.
    Returns:
      The prediction of the output tensor.
    """
    # adds one column ?
    states = np.expand_dims(states, axis=0)
    # tell Tensorflow to predict
    # results in a tensor of 3 values containing ... what ?
    prediction = sess.run(self.y_est, {self.X: states})
    #print(prediction)
    return prediction

  def update(self, sess, states, actions, targets):
    """
    Updates the weights of the neural network, based on its targets, its
    predictions, its loss and its optimizer.
    
    Args:
      sess: TensorFlow session.
      states: [current_state] or states of batch
      actions: [current_action] or actions of batch
      targets: [current_target] or targets of batch
    """
    updates = np.zeros((1,3)) 
    updates = self.predict(sess, states)
    updates[0][actions] = targets
    states = np.expand_dims(states, axis=0)
    sess.run(self.train, feed_dict={self.X: states, self.y: updates})
    return 0 

class TargetNetwork(NeuralNetwork):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """
    def __init__(self, tau=0.001):
      NeuralNetwork.__init__(self)
      self.tau = tau
      self._associate = self._register_associate()

    def _register_associate(self):
      tf_vars = tf.trainable_variables()
      total_vars = len(tf_vars)
      op_holder = []
      for idx,var in enumerate(tf_vars[0:total_vars//2]):
          op_holder.append(tf_vars[idx+total_vars//2].assign((var.value
            ()*self.tau) + ((1-self.tau)*tf_vars[idx+total_vars//2].value())))
      return op_holder
        
    def update(self, sess):
      for op in self._associate:
        sess.run(op)

class ReplayBuffer:
  #Replay buffer for experience replay. Stores transitions.
  def __init__(self):
    self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "dones"])
    self._data = self._data(states=[], actions=[], next_states=[], rewards=[], dones=[])

  def add_transition(self, state, action, next_state, reward, done):
    self._data.states.append(state)
    self._data.actions.append(action)
    self._data.next_states.append(next_state)
    self._data.rewards.append(reward)
    self._data.dones.append(done)

  def next_batch(self, batch_size):
    batch_indices = np.random.choice(len(self._data.states), batch_size)
    batch_states = np.array([self._data.states[i] for i in batch_indices])
    batch_actions = np.array([self._data.actions[i] for i in batch_indices])
    batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
    batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
    batch_dones = np.array([self._data.dones[i] for i in batch_indices])
    return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones

def make_epsilon_greedy_policy(estimator, epsilon, nA):
  """
  Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
  
  Args:
      estimator: An estimator that returns q values for a given state
      epsilon: The probability to select a random action . float between 0 and 1.
      nA: Number of actions in the environment.
  
  Returns:
      A function that takes the observation as an argument and returns
      the probabilities for each action in the form of a numpy array of length nA.
  
  """
  def policy_fn(sess, observation):
    A = np.ones(nA, dtype=float) * epsilon / nA
    q_values = estimator.predict(sess, observation)
    best_action = np.argmax(q_values)
    A[best_action] += (1.0 - epsilon)
    return A
  return policy_fn

def q_learning(sess, env, approx, num_episodes, max_time_per_episode,
        discount_factor=0.99, epsilon=0.1, use_experience_replay=False, batch_size=128, target=None):
  """
  Q-Learning algorithm for off-policy TD control using Function Approximation.
  Finds the optimal greedy policy while following an epsilon-greedy policy.
  Implements the options of online learning or using experience replay and also
  target calculation by target networks, depending on the flags. You can reuse
  your Q-learning implementation of the last exercise.

  Args:
    env: OpenAI environment.
    approx: Action-Value function estimator
    num_episodes: Number of episodes to run for.
    max_time_per_episode: maximum number of time steps before episode is terminated
    discount_factor: gamma, discount factor of future rewards.
    epsilon: Chance to sample a random action. Float betwen 0 and 1.
    use_experience_replay: Indicator if experience replay should be used.
    batch_size: Number of samples per batch.
    target: Slowly updated target network to calculate the targets. Ignored if None.

  Returns:
    An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
  """
  # Replay Experiences by Buffer
  experience = ReplayBuffer()
  batches = []
  batchCount = 0
  batchLearning = False
  # Keeps track of useful statistics
  stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))    

  # for episode=1...M do
  for i_episode in range(num_episodes):
      
      # seleect epsolon-greedy action  
      policy = make_epsilon_greedy_policy(
          approx, epsilon, env.action_space.n)
      
      # Print out which episode we're on, useful for debugging.
      # Also print reward for last episode
      last_reward = stats.episode_rewards[i_episode - 1]
      print("\rEpisode {}/{} ({}) [Batch Learning: {}]".format(i_episode + 1, num_episodes, last_reward,batchLearning))
      sys.stdout.flush()
      
      state = env.reset()
      
      terminal_flag = False
      current_run_buffer = []
      
      if(batchCount > batch_size):
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = random.choice(batches)
      for t in range(max_time_per_episode):

          if(batchCount > batch_size):
            #batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = randBatch
            batchLearning = True
          if batchLearning:
            state = batch_states[t%batch_size]    
          
          if not batchLearning:
            action_probs = policy(sess, state)
            action = np.random.choice(3, p=action_probs)
            
          else:
            action = batch_actions[t%batch_size]

          # observer next_reward. next_state and terminal_flag
          if batchLearning:
            next_state, reward, done=  batch_next_states[t%batch_size], batch_rewards[t%batch_size], batch_dones[t%batch_size]
          else:
            next_state, reward, done, info = env.step(action)
          stats.episode_rewards[i_episode] += reward
          stats.episode_lengths[i_episode] = t
          td_target = reward

          # store (s,a,s_t+1,r_t+q,terminal_flag) in replay buffer
          current_run_buffer.append([state, action,next_state,reward,done])

          # if not terminal_flag y+1 = r_t+1+discount*maxQ
          if not done:
            if ((not use_experience_replay) or (not batchLearning)):
              td_target += discount_factor * np.max(approx.predict(sess, next_state))
              # update Q on squared difference of sample
              approx.update(sess, state, action, td_target) 
            else:
              td_target += discount_factor * np.max(target.predict(sess, next_state))
               # update Q on mean squared error (MSE)
               # amd shift weights of target Q
              target.update(sess) 

          # if terminal_flag in this state y = r_t+1
          if done:
              terminal_flag = True
              # store batch
              batchCount += 1
              print("Storing {} (of min {} wanted) full successfully DONE episode to batches!".format(batchCount,batch_size))
              for e in current_run_buffer:
                experience.add_transition(e[0],e[1],e[2],e[3],e[4])
              nb = experience.next_batch(batch_size)
              batches.append(nb)
              
              break
          
          state = next_state
  return stats

def plot_episode_stats(stats, smoothing_window=10, noshow=True):
  # Plot the episode length over time
  fig1 = plt.figure(figsize=(10,5))
  plt.plot(stats.episode_lengths)
  plt.xlabel("Episode")
  plt.ylabel("Episode Length")
  plt.title("Episode Length over Time")
  fig1.savefig('episode_lengths.png')
  if noshow:
      plt.close(fig1)
  else:
      plt.show(fig1)

  # Plot the episode reward over time
  fig2 = plt.figure(figsize=(10,5))
  rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
  plt.plot(rewards_smoothed)
  plt.xlabel("Episode")
  plt.ylabel("Episode Reward (Smoothed)")
  plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
  fig2.savefig('reward.png')
  if noshow:
      plt.close(fig2)
  else:
      plt.show(fig2)

if __name__ == "__main__":
  env = MountainCarEnv()
  approx = NeuralNetwork()
  target = TargetNetwork()
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  # Choose one.
  #stats = q_learning(sess, env, approx, 3000, 1000)
  stats = q_learning(sess, env, approx, 1000, 1000, use_experience_replay=True, batch_size=128, target=target)
  plot_episode_stats(stats)

  for _ in range(100):
    state = env.reset()
    for _ in range(1000):
      env.render()
      state,_,done,_ = env.step(np.argmax(approx.predict(sess, state)))
      if done:
        break