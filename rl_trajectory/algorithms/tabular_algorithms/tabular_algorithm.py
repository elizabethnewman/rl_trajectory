import numpy as np
import gym
from typing import Tuple
from copy import deepcopy
from rl_trajectory.algorithms.rl_algorithm import RLAlgorithm
from rl_trajectory.policies.tabular_policies import TabularPolicy, TabularEpsilonGreedy


class TabularLearning(RLAlgorithm):

    def __init__(self, policy: TabularPolicy = TabularEpsilonGreedy(), **kwargs):
        super(TabularLearning, self).__init__(policy=policy, **kwargs)

        if self.store_steps:
            self.info = {'ieps': [], 'eps': [], 'state': [], 'reward': [], 'Q_list': [], 'S_list': [], 'S_total_list': []}

        self.state_info = dict()

        # algorithm history
        self.history = {'header': ('episode', 'sum|Q|', 'eps', 'last_action', 'last_reward',
                                   'total_reward', 'avg_reward', 'num_steps', 'flag'),
                        'format': '{:<15d}{:<15.2e}{:<15.2e}{:<15d}{:<15.2f}{:<15.2f}{:<15.2f}{:<15d}{:<15d}',
                        'value': []}
        self.history['value'] = np.empty((0, len(self.history['header'])))

    def __repr__(self):
        return "EpsilonGreedyTabularLearning(gamma=%1.1f,alpha=%1.1f,num_episodes=%d,max(eps)=%1.1e, min(eps)=%1.2e)" \
               % (self.gamma, self.alpha, self.num_episodes, self.max_eps, self.min_eps)

    def reset(self):
        return None

    def train(self, env: gym.Env, Q, verbose: bool = False, log_interval: int = 1) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        # ------------------------------------------------------------------------------------------------------------ #
        # size of action and state space
        grid_size = Q.shape[:-1]
        num_actions = Q.shape[-1]

        # number of state-action visits
        # R = np.zeros(tuple(grid_size) + (num_actions,), dtype=np.compat.long)
        self.state_info = {'S': np.zeros(tuple(grid_size) + (num_actions,), dtype=np.compat.long),
                           'S_total': np.zeros(tuple(grid_size))}

        self.best = {'S': deepcopy(self.state_info['S']),
                     'S_total': deepcopy(self.state_info['S_total']),
                     'Q': deepcopy(Q), 'ieps': 0, 'total_reward': -np.inf}

        # print headers
        if verbose:
            print(('{:<15s}' * len(self.history['header'])).format(*self.history['header']))

        # ------------------------------------------------------------------------------------------------------------ #
        # main iteration
        for ieps in range(self.num_episodes):

            info = self.run_episode(env, Q, ieps)

            last_action = info['last_action']
            last_reward = info['last_reward']
            total_reward = info['total_reward']
            termination_flag = info['termination_flag']

            # -------------------------------------------------------------------------------------------------------- #
            # store relevant history per episode
            history_iter = [ieps, sum(abs(Q.reshape(-1))), self.eps[ieps], last_action, last_reward,
                            total_reward, total_reward / self.num_steps, self.num_steps, termination_flag]

            if total_reward >= self.best['total_reward'] and termination_flag == 1:
                self.best['S'] = np.copy(self.state_info['S'])
                self.best['S_total'] = np.copy(self.state_info['S_total'])
                self.best['Q'] = np.copy(Q)
                self.best['total_reward'] = total_reward

            self.history['value'] = np.concatenate((self.history['value'], np.array(history_iter).reshape(1, -1)))
            if ieps % log_interval == 0:
                print(self.history['format'].format(*history_iter))

        results = {'Q': Q, 'best': self.best, 'history': self.history}
        return results

    def run_episode(self, env, Q, ieps):
        raise NotImplementedError

    def update_Q(self, Q: np.ndarray, action: int, reward: float, done: bool,
                 state_dis: np.ndarray, next_state_dis: np.ndarray, eps: float) -> (np.ndarray, int):
        raise NotImplementedError

    def update_info(self, state, reward, ieps, Q):
        if self.store_steps and ((ieps + 1) % self.store_every == 0 or ieps == self.num_episodes - 1 or ieps == 0):
            self.info['ieps'].append(ieps)
            self.info['eps'].append(self.eps[ieps])
            self.info['state'].append(deepcopy(state).reshape(1, -1))
            self.info['reward'].append(deepcopy(reward))
            self.info['S_list'].append(deepcopy(self.state_info['S']))
            self.info['S_total_list'].append(deepcopy(self.state_info['S_total']))
            self.info['Q_list'].append(deepcopy(Q))


