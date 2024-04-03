import numpy as np
from torch import Tensor
import torch.nn as nn
from typing import Union


class Featurizer(nn.Module):

    def __init__(self):
        super(Featurizer, self).__init__()

    def num_features(self, num_state: int) -> int:
        raise NotImplementedError

    def forward(self, state: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        raise NotImplementedError


class ConcatenatedFeaturizer(nn.Module):

    def __init__(self, *args):
        super(ConcatenatedFeaturizer, self).__init__()
        self.featurizers = []
        for feat in args:
            self.featurizers.append(feat)

    def num_features(self, num_state: int) -> int:
        n = 0
        for feat in self.featurizers:
            n += feat.num_features(num_state)
        return n

    def forward(self, state: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        state_feat = np.empty(0)
        for feat in self.featurizers:
            state_feat = np.concatenate((state_feat, feat(state)))

        return state_feat
