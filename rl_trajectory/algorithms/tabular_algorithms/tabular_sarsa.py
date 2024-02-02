from rl_trajectory.algorithms.tabular_algorithms.tabular_algorithm import TabularLearning
from rl_trajectory.utils import discretize_box_state
import numpy as np


class TabularSARSA(TabularLearning):

    def __init__(self, *args, **kwargs):
        super(TabularSARSA, self).__init__(*args, **kwargs)

    def run_episode(self, env, Q, ieps):
        reward, done, total_reward, termination_flag, action = self.initialize_episode()

        grid_size = np.array(Q.shape[:-1]).astype(np.int8)

        # reset environment
        state = np.copy(env.reset()[0])
        self.update_info(state, reward, ieps, Q)

        while not done:

            # ---------------------------------------------------------------------------------------------------- #
            # discretize state
            state_dis = discretize_box_state(state, grid_size, env.observation_space)

            # choose action
            action = self.policy(Q, state_dis, self.eps[ieps])

            # step
            next_state, reward, done, _, info = env.step(action)

            # TODO: only discretize if not done
            next_state_dis = discretize_box_state(next_state, grid_size, env.observation_space)

            # ---------------------------------------------------------------------------------------------------- #
            # algorithm termination - too many steps or outside observation space
            reward, done, termination_flag = self.episode_termination_conditions(env, next_state, reward, done)

            # ---------------------------------------------------------------------------------------------------- #
            if done:
                Q[tuple(state_dis) + (action,)] = reward
                self.state_info['S_total'][tuple(next_state_dis)] += 1  # TODO: why is this here?
            else:
                # Sutton & Barto, Section 6.5
                # Q(state, action) += alpha * [reward + gamma * Q(next_state, a') - Q(state, action)]
                # update
                next_action = self.policy(Q, next_state_dis, self.eps[ieps])

                Q[tuple(state_dis) + (action,)] += \
                    self.alpha * (reward + self.gamma * Q[tuple(next_state_dis) + (next_action,)]
                                  - Q[tuple(state_dis) + (action,)])

                state = np.copy(next_state)
                action = np.copy(next_action)

            # ---------------------------------------------------------------------------------------------------- #

            # update S
            self.state_info['S'][tuple(state_dis) + (action,)] += 1
            self.state_info['S_total'][tuple(state_dis)] += 1

            self.update_info(state, reward, ieps, Q)

            # update steps
            total_reward += reward
            self.num_steps += 1

        info = dict()
        info['last_action'] = action
        info['last_reward'] = reward
        info['total_reward'] = total_reward
        info['termination_flag'] = termination_flag
        return info


if __name__ == "__main__":
    import rl_trajectory.envs as env
    import gym
    from rl_trajectory.utils import seed_everything

    traj = env.TrajectoryEnv(grid_size=np.array((16, 16)))
    traj.x0 = gym.spaces.Box(low=np.array([-1, -1], dtype=np.float32),
                             high=np.array([1, 1], dtype=np.float32),
                             dtype=np.float32)

    seed_everything(42, traj)

    opt = TabularSARSA(max_steps=100, num_episodes=10, penalty=0, max_eps=1e0, min_eps=5e-3, alpha=0.5)

    Q = np.zeros(tuple(traj.grid_size) + (traj.action_space.n,), dtype=np.float32)
    results = opt.train(traj, Q, verbose=True, log_interval=1)
