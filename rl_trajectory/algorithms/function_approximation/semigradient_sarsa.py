import numpy as np
import torch
from rl_trajectory.algorithms.function_approximation.function_approximation_algorithm import LinearFunctionApproximationAlgorithm
import math
import gym
from copy import deepcopy
from rl_trajectory.utils import extract_data, insert_data


class LinearSemiGradientSARSA(LinearFunctionApproximationAlgorithm):
    r"""
    Episodic Semi-Gradient SARSA
    Sutton & Barto, page 244
    http://incompleteideas.net/book/RLbook2020.pdf
    """

    def __init__(self, *args, **kwargs):
        super(LinearSemiGradientSARSA, self).__init__(*args, **kwargs)
        self.history['header'] += ('iter', 'alpha', '|w - w_old|')
        self.history['format'] += '{:<15d}{:<15.2e}{:<15.2e}'

    def run_episode(self, env, model, ieps):

        reward, done, total_reward, termination_flag, action = self.initialize_episode()

        eps = self.eps[ieps]

        # store weights before running episode
        w_old = torch.clone(model.w.data)

        # initialize state and action
        state = np.copy(env.reset()[0])

        # choose action
        action = self.policy(model, state, eps)

        if self.store_steps and ((ieps + 1) % self.store_every == 0 or ieps == self.num_episodes - 1 or ieps == 0):
            self.step_info['w_list'][ieps] = torch.clone(model.w.data)



        while not done:
            # step
            next_state, reward, done, _, _ = env.step(action)

            # ---------------------------------------------------------------------------------------------------- #
            # algorithm termination - too many steps or outside observation space
            reward, done, termination_flag = self.episode_termination_conditions(env, next_state, reward, done)

            # ---------------------------------------------------------------------------------------------------- #
            # update step size
            if not self.constant_step:
                alpha = self.alpha / (1.0 + math.sqrt(self.iter_count + 1))
            else:
                alpha = self.alpha

            # form gradient now
            theta = extract_data(model, 'data')
            q, q_grad = self.linearize(model, state, action)

            # compute Bellman error before next state
            D = reward - q

            if not done:
                # update Bellman error with next state information
                next_action = self.policy(model, next_state, eps)
                D += self.gamma * model(next_state, next_action)

                # no additional gradient information (semi-gradient)

                state = deepcopy(next_state)
                action = deepcopy(next_action)
                self.iter_count += 1

            # update weights
            dtheta = -q_grad * D
            insert_data(model, theta.view(-1) - alpha * dtheta.view(-1))

            if done:
                self.extra_info = [self.iter_count, alpha, torch.norm(model.w.data - w_old)]

            # update steps
            total_reward += reward
            self.num_steps += 1

        # get info, stored as dict
        info = self.episode_info(total_reward, action, reward, termination_flag)
        return info


if __name__ == "__main__":
    import rl_trajectory.envs as env
    from rl_trajectory.featurizers import PolynomialFeaturizer
    import rl_trajectory.models as models
    import gym

    traj = env.TrajectoryEnv(grid_size=np.array((16, 16)))
    traj.x0 = gym.spaces.Box(low=np.array([-1, -1], dtype=np.float32),
                             high=np.array([1, 1], dtype=np.float32),
                             dtype=np.float32)

    featurizer = PolynomialFeaturizer(order=2)

    num_actions = traj.action_space.n
    num_state = traj.observation_space.shape[0]
    num_features = featurizer.num_features(num_state)
    model = models.LinearModel(featurizer, num_state, num_actions)

    opt = LinearSemiGradientSARSA(num_episodes=10, max_steps=100, alpha=1e-2,
                                  beta=1.5e-5, gamma=0.9, lambda0=1e-4, constant_step=False)
    results = opt.train(traj, model, verbose=True, log_interval=1)

