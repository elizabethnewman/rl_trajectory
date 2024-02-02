import numpy as np
import gym


def discretize_box_state(state: np.ndarray, grid_size: np.ndarray, observation_space: gym.spaces.Box or np.array) -> np.ndarray:
    """
    Discretize state based on domain
    If x \in [low + i * h, low + (i + 1) * h), then x is at grid point i
    Inputs:
        state : state on continuous domain
        grid_size : number of cells per dimension
        observation_space : domain
    """
    if isinstance(observation_space, gym.spaces.Box):
        low = observation_space.low
        high = observation_space.high
    else:
        # low = np.array([observation_space[0], observation_space[2]])
        # high = np.array([observation_space[1], observation_space[3]])
        low = observation_space[0::2]
        high = observation_space[1::2]

    # find step size
    h = (high - low) / grid_size

    # discretize
    state_dis = np.floor((state - low) / h).astype(np.int8)

    # make sure we do not leave grid
    state_dis = np.minimum(grid_size - 1, np.maximum(0, state_dis))

    return state_dis
