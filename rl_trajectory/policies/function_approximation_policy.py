import numpy as np
import torch


class FunctionApproximationPolicy(torch.nn.Module):

    def __init__(self):
        super(FunctionApproximationPolicy, self).__init__()

    def forward(self, model, state_feat, *args, **kwargs):
        raise NotImplementedError


class FunctionApproximationGreedy(FunctionApproximationPolicy):

    def __init__(self):
        super(FunctionApproximationGreedy, self).__init__()

    def forward(self, model, state, *args, **kwargs):

        if isinstance(model, torch.Tensor):
            values = model.view(-1, state.numel()) @ state
        else:
            values = model(state)

        # action = np.random.choice([act for act, value in enumerate(values) if value == max(values)])
        actions = [act for act, value in enumerate(values) if value == values.max()]
        action = actions[torch.randperm(len(actions))[0]]
        return action


class FunctionApproximationEpsilonGreedy(FunctionApproximationPolicy):

    def __init__(self):
        super(FunctionApproximationEpsilonGreedy, self).__init__()
        self.greedy_policy = FunctionApproximationGreedy()

    def forward(self, model, state, *args, **kwargs):
        eps = args[0]
        if np.random.random() > eps:
            action = self.greedy_policy(model, state, *args, **kwargs)
        else:
            # exploration
            action = np.random.randint(model.num_actions)

        return action
