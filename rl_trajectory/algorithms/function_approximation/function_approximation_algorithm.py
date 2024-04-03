import torch
import numpy as np
import gym
import logging
from typing import Tuple
from rl_trajectory.policies import FunctionApproximationPolicy, FunctionApproximationEpsilonGreedy
from rl_trajectory.models import Model, LinearModel
from rl_trajectory.featurizers import Featurizer
from rl_trajectory.utils import set_seed, choose_exploration_parameters
from rl_trajectory.algorithms.rl_algorithm import RLAlgorithm
from rl_trajectory.utils import extract_data, seed_everything

import torch.func as func


class LinearFunctionApproximationAlgorithm(RLAlgorithm):
    r"""
    Only for linear models for now
    """

    def __init__(self, policy: FunctionApproximationPolicy = FunctionApproximationEpsilonGreedy(), **kwargs):
        super(LinearFunctionApproximationAlgorithm, self).__init__(policy=policy, **kwargs)

        self.step_info = {'w_list': {}}

        # algorithm history
        self.history = {'header': ('episode', 'sum|w|', 'eps', 'gammma', 'last_action', 'last_reward',
                                   'total_reward', 'avg_reward', 'num_steps', 'flag'),
                        'format': '{:<15d}{:<15.2e}{:<15.2e}{:<15.2f}{:<15d}{:<15.2f}{:<15.2f}{:<15.2f}{:<15d}{:<15d}',
                        'value': []}

        # self.history['value'] = np.empty((0, len(self.history['header'])))
        self.extra_info = []
        self.info = {'eigH': []}
        self.best = None

    def __repr__(self):
        return "LinearFunctionApproxmationLearning(gamma=%1.1f,alpha=%1.1f,num_episodes=%d,max(eps)=%1.1e, min(eps)=%1.2e)" \
               % (self.gamma, self.alpha, self.num_episodes, self.max_eps, self.min_eps)

    def linearize(self, model, state, action):
        model.zero_grad()
        q = model(state, action)
        q.backward()
        q_grad = extract_data(model, 'grad').view(-1, 1)
        return q, q_grad

    def train(self, env: gym.Env, model: LinearModel,
              verbose: bool = False, log_interval: int = 1, logger: logging.RootLogger = None) -> dict:
    # Instead of passing logger to train, we can also get logger by its name

        if self.seed is not None:
            seed_everything(self.seed, env)

        self.best = {'w': torch.clone(model.w.data), 'ieps': 0, 'total_reward': -np.inf, 'num_steps': 1}
        # ------------------------------------------------------------------------------------------------------------ #
        # print headers
        if verbose:
            if logger is not None:
                logger.info(('{:<15s}' * (len(self.history['header']))).format(*(self.history['header'])))
            else:
                print(('{:<15s}' * (len(self.history['header']))).format(*(self.history['header'])))

        # state = np.copy(env.reset()) # for comparison to original code
        # ------------------------------------------------------------------------------------------------------------ #
        # main iteration
        for ieps in range(self.num_episodes):

            episode_info = self.run_episode(env, model, ieps)

            total_reward = episode_info['total_reward']
            last_action = episode_info['last_action']
            last_reward = episode_info['last_reward']
            termination_flag = episode_info['termination_flag']

            # -------------------------------------------------------------------------------------------------------- #
            # store relevant history per episode
            sum_theta = torch.sum(torch.abs(model.w.data)).item()
            history_iter = ([ieps, sum_theta, self.eps[ieps], self.gamma, last_action, last_reward,
                            total_reward, total_reward / self.num_steps, self.num_steps, termination_flag]
                            + self.extra_info)

            # best based on average reward
            if total_reward / self.num_steps >= self.best['total_reward'] / self.best['num_steps'] and termination_flag == 1:
                self.best['w'] = torch.clone(model.w.data)
                self.best['ieps'] = ieps
                self.best['total_reward'] = total_reward
                self.best['num_steps'] = self.num_steps

            self.history['value'].append(history_iter)
            if verbose and ieps % log_interval == 0:
                if logger is not None:
                    logger.info((self.history['format']).format(*history_iter))
                else:
                    print((self.history['format']).format(*history_iter))

        results = {'history': self.history, 'best': self.best, 'info': self.info}
        return results

    def run_episode(self, env, model, ieps):
        raise NotImplementedError

    def episode_info(self, total_reward, last_action, last_reward, termination_flag):
        info = dict()
        info['total_reward'] = total_reward
        info['last_action'] = last_action
        info['last_reward'] = last_reward
        info['termination_flag'] = termination_flag
        return info









