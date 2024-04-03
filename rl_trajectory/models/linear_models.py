import numpy as np
import torch
import torch.nn as nn
from rl_trajectory.featurizers import Featurizer, PolynomialFeaturizer
from typing import Union


class Model(nn.Module):
    def __init__(self, featurizer: Featurizer):
        super(Model, self).__init__()
        self.featurizer = featurizer

    def forward(self, state_feat: torch.Tensor, action: int = None):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class LinearModel(Model):
    # TODO: long term - make more efficient for coarse coding
    # NOTE: not compatible with PyTorch optimizers
    def __init__(self, featurizer: Featurizer, num_state: int, num_actions: int):
        super(LinearModel, self).__init__(featurizer)
        self.num_state = num_state
        self.num_features = featurizer.num_features(num_state)
        self.num_actions = num_actions
        self.w = nn.Parameter(torch.zeros(self.num_features, num_actions))
        self.ctx = None

    def forward(self, state: np.ndarray, action: int = None) -> torch.Tensor:
        state_feat = torch.tensor(self.featurizer(state), device=self.w.device, dtype=self.w.dtype)
        self.ctx = (state_feat, action)

        if action is None:
            values = state_feat @ self.w
        else:
            values = state_feat @ self.w[:, action]

        return values

    def backward(self):
        state_feat, action = self.ctx
        grad_w = None

        if action is not None:
            # ei = torch.zeros(self.num_actions, dtype=state_feat.dtype, device=state_feat.device)
            # ei[action] = 1.0
            # state_feat = torch.kron(ei.view(-1), state_feat.view(-1))
            grad_w = torch.zeros_like(self.w)
            grad_w[:, action] = state_feat

        return grad_w



