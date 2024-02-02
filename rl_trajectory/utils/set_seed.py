import torch
import numpy as np
import random
import gym
from gym.utils.seeding import np_random


def seed_everything(seed: int = 42, env: gym.Env = None):
    # create_seed(seed)
    rng, seed = np_random(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

    if env is not None:
        # env.seed(seed)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        # env.action_space.np_random.seed(seed)
        # env.observation_space.np_random.seed(seed)
        env.observation_space.seed(seed)
