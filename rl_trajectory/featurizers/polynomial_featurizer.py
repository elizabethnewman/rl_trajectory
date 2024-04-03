import numpy as np
import math
from torch import Tensor
from typing import Union
from rl_trajectory.featurizers import Featurizer


class PolynomialFeaturizer(Featurizer):

    def __init__(self, order: int = 2):
        super(PolynomialFeaturizer, self).__init__()
        if not isinstance(order, int):
            raise ValueError("order must be integer")
        # if order < 0:
        #     raise ValueError("order must be greater or equal to 0")
        # if order >= 4:
        #     raise NotImplementedError("order 4 or higher not yet implemented")

        self.order = order

    def num_features(self, num_state: int) -> int:
        return dim_features(num_state, self.order)

    def forward(self, state: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        return degree_recursive(state.reshape(-1), self.order)


def dim_features(dim_state, degree):
    # https://math.stackexchange.com/questions/36250/number-of-monomials-of-certain-degree
    if degree == 0:
        return 1
    else:
        n = int(math.factorial(dim_state + degree - 1) / (math.factorial(degree) * math.factorial(dim_state - 1)))
        return n + dim_features(dim_state, degree - 1)


def degree_recursive(state, degree):

    if degree == 0:
        return np.ones(1, dtype=state.dtype)
    else:
        fv = degree_recursive(state, degree - 1)

        if degree == 1:
            return np.concatenate((fv, state))
        else:
            # number of features for first state coordinate
            num_features_stata_a = dim_features(len(state), degree - 1) - dim_features(len(state), degree - 2)

            fv_past = np.copy(fv)
            count = dim_features(len(state), degree - 2)
            for i, s in enumerate(state):
                tmp = s * fv_past[count:]
                fv = np.concatenate((fv, tmp))

                if i == len(state) - 1:
                    num_features_state_b = 0
                else:
                    num_features_state_b = (dim_features(len(state) - i - 1, degree - 1) -
                                            dim_features(len(state) - i - 1, degree - 2))

                count += num_features_stata_a - num_features_state_b

                num_features_stata_a = np.copy(num_features_state_b)

            return fv


if __name__ == "__main__":

    s = np.array([2, 3, 5, 7, 11])
    dim_s = len(s)

    degree = 3
    fv = degree_recursive(s, degree)
    print(len(fv), fv)
    print(dim_features(dim_s, degree))






