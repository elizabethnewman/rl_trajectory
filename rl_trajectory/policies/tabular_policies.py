import torch.nn as nn
import numpy as np


class TabularPolicy(nn.Module):

    def __init__(self):
        super(TabularPolicy, self).__init__()

    def forward(self, Q: np.ndarray, state: np.ndarray, *args, **kwargs):
        raise NotImplementedError


class TabularGreedy(TabularPolicy):

    def __init__(self):
        super(TabularGreedy, self).__init__()

    def forward(self, Q: np.ndarray, state: np.ndarray, *args, **kwargs):
        values = Q[tuple(state)]
        action = np.random.choice([act for act, value in enumerate(values) if value == values.max()])
        return action


class TabularEpsilonGreedy(TabularPolicy):

    def __init__(self):
        super(TabularEpsilonGreedy, self).__init__()
        self.greedy_policy = TabularGreedy()

    def forward(self, Q: np.ndarray, state: np.ndarray, *args, **kwargs):
        eps = args[0]
        if np.random.random() > eps:
            action = self.greedy_policy(Q, state, *args, **kwargs)
        else:
            # exploration
            num_actions = Q.shape[-1]
            action = np.random.randint(num_actions)

        return action
