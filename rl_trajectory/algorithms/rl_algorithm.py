import torch
import numpy as np
import gym
from typing import Tuple
from rl_trajectory.utils import set_seed, choose_exploration_parameters
from typing import Union
from rl_trajectory.policies import TabularPolicy


class RLAlgorithm:
    r"""
    Only for linear models for now
    """

    def __init__(self,
                 policy: TabularPolicy = None,
                 gamma: float = 1.0,
                 alpha: float = 0.01,
                 num_episodes: int = 1000,
                 max_eps: float = 1e-1,
                 min_eps: float = 1e-3,
                 max_eta: float = 1e-2,
                 min_eta: float = 1e-8,
                 max_steps: int = 100,
                 penalty: float = 0.0,
                 constant_step: bool = True,
                 store_steps: bool = False,
                 store_every: int = 1,
                 seed: int = None, **kwargs):
        super(RLAlgorithm, self).__init__()
        self.seed = seed                        # random seed
        self.policy = policy                    # policy to choose actions
        self.gamma = gamma                      # discount factor
        self.alpha = alpha                      # step size
        self.constant_step = constant_step      # decay or constant step size
        self.num_episodes = num_episodes        # number of episodes
        self.max_steps = max_steps              # maximum number of steps per episode
        self.max_eps = max_eps                  # most exploration allowed (eps = 1 --> random guess)
        self.min_eps = min_eps                  # least exploration allowed (eps = 0 --> greedy)
        self.penalty = penalty                  # penalty for failed episode based on number of steps

        # exploration parameters per episode
        self.eps = np.logspace(np.log10(max_eps), np.log10(min_eps), num=num_episodes)
        self.eta = np.logspace(np.log10(min_eta), np.log10(max_eta), num=num_episodes)

        self.iter_count = 0                     # counting total iterations over all episodes
        self.num_steps = 0                      # counting steps per episode (reset to 0 at the beginning of each episode)

        self.store_steps = store_steps          # flag to store steps in each episode
        self.store_every = store_every          # store steps every n steps

        # algorithm history, defined per algorithm
        self.history = {'header': (), 'format': '', 'value': []}
        self.extra_info = []
        self.info = None

        # place to store info from the best model so far
        self.best = None

    def __repr__(self):
        raise NotImplementedError

    def train(self, env: gym.Env, model: np.ndarray,
              verbose: bool = False, log_interval: int = 1) -> dict:
        r"""
        Train the model using the algorithm, which will differ for Tabular and Function Approximation algorithms
        """
        raise NotImplementedError

    def run_episode(self, env, model, ieps):
        r"""
        Run an episode of the algorithm, which is the primary difference between all algorithms
        """
        raise NotImplementedError

    def episode_termination_conditions(self, env, next_state, reward, done):
        r"""
        Termination conditions for the episode (e.g. max steps, leaving domain, etc.)
        """
        termination_flag = None
        if done:
            termination_flag = 1

        # take too many steps
        if self.num_steps >= self.max_steps:
            reward -= self.penalty
            termination_flag = -1
            done = True

        # leave domain
        if isinstance(env.observation_space, gym.spaces.Box) and not env.observation_space.contains(next_state):
            reward -= self.penalty
            termination_flag = -2
            done = True
        return reward, done, termination_flag

    def episode_info(self, total_reward, last_action, last_reward, termination_flag):
        r"""
        Information to be stored in the history at the end of each episode

        This function is used for convenience and called in run_episode()
        """
        info = dict()
        info['total_reward'] = total_reward
        info['last_action'] = last_action
        info['last_reward'] = last_reward
        info['termination_flag'] = termination_flag
        return info

    def initialize_episode(self):
        reward = 0
        done = False
        total_reward = 0
        termination_flag = None
        action = None
        self.num_steps = 0
        return reward, done, total_reward, termination_flag, action
