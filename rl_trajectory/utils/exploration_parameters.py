import numpy as np


def choose_exploration_parameters(range_eps: tuple = (1.0, 1e-4), num_episodes: tuple = (100,)):

    assert len(range_eps) == len(num_episodes) + 1

    # sort range
    range_eps = np.sort(range_eps)
    range_eps = range_eps[::-1]

    eps = np.empty(0)
    for i, n in enumerate(num_episodes):
        tmp_eps = np.logspace(np.log10(range_eps[i]), np.log10(range_eps[i + 1]), num=n)

        if i > 0:
            eps = np.concatenate((eps, tmp_eps[1:]))
        else:
            eps = np.concatenate((eps, tmp_eps))

    return eps


if __name__ == "__main__":
    eps = choose_exploration_parameters((1, 1e-4), (5,))
    print(eps)

    eps = choose_exploration_parameters((1, 1e-2, 1e-4), (2, 2))
    print(eps)