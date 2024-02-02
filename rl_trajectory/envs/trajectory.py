import numpy as np
import gym
from gym import spaces
from rl_trajectory.utils import discretize_box_state
from typing import Optional
from gym.utils import seeding
from copy import deepcopy
import matplotlib.pyplot as plt


class TrajectoryEnv(gym.Env):
    """Simple environment for a trajectory problem

    Description:

        The goal is to move efficiently from a fixed starting point to a target.
        The state is a vector x, which contains the current position and there
        are four actions (left,right,up,down) available.
        The cost function is
        # should depend on hx nad hy instead of -1
        J(u) = \sum(-1 - Q(x))

    Observation:

        Type: Box(2)
        Num   Observation                     Min       Max
        0     x_1 coordinate                 -1.0       1.0
        1     x_2 coordinate                 -1.0       1.0

    Actions:

        Type: Discrete(4)
        Num  Action
        0    increase x_1 by h_x (move right)
        1    increase x_2 by h_y (move up)
        2    decrease x_1 by h_x (move left)
        3    decrease x_2 by h_y (move down)

    Reward:
        - 1 - Q(x), where is a standard Gaussian that increases the costs of moving around
        the origin.

    Starting State:
        Each episode begins by sampling from the user-specified starting state gym environment x0.

    Episode Termination:
        Either when target is reached or maximum number of steps (default=100) is reached
    """

    def __init__(self,
                 x0: gym.Space = None,
                 x_target: np.ndarray = None,
                 grid_size: np.ndarray = None,
                 low: np.ndarray = None,
                 high: np.ndarray = None,
                 dtype: np.dtype = np.float32,
                 max_height: float = 1.0,
                 slope: float = 1.0,
                 max_step_count: int = 2000):
        # super(TrajectoryEnv, self).__init__()

        # declare action space (left, right, down, up)
        self.action_space = spaces.Discrete(4)

        # declare observation space
        if low is None:
            low = -np.ones(2, dtype=dtype)
        if high is None:
            high = np.ones(2, dtype=dtype)

        # grid size (the number of cell centers)
        if grid_size is None:
            grid_size = np.array([8, 8], dtype=np.int8)  # number of cell centers
        self.grid_size = grid_size
        self.h = (high - low) / grid_size  # step size between cell centers

        # allow observations in any part of cell
        self.observation_space = spaces.Box(low, high, dtype=dtype)

        # domain information (redundancy)
        self.low = low
        self.high = high
        self.domain = np.array([low[0], high[0], low[1], high[1]], dtype=np.float32)

        # starting state space
        if x0 is None:
            s0 = [low[0] + 1.5 * self.h[0], low[1] + 1.5 * self.h[1]]
            x0 = spaces.Box(np.array(s0, dtype=dtype), np.array(s0, dtype=dtype), dtype=dtype)  # only sample one point
        self.x0 = x0

        if x_target is None:
            x_target = np.array([high[0] - 1.5 * self.h[0], high[1] - 1.5 * self.h[1]], dtype=dtype)

        # target (adjusted for grid)
        self.x_target_dis = self.state2grid(x_target)
        self.x_target = self.grid2state(self.x_target_dis)

        # current state
        self.state = None

        # describe terrain
        self.max_height = max_height
        self.slope = slope
        self.step_count = 0
        self.max_step_count = max_step_count

        # visualization
        self.viewer = None

        # store info
        self.info = {}

    def __repr__(self):
        return "Goal: Find trajectory to [%0.3f, %0.3f] in fewer than 100 steps." % (self.x_target[0], self.x_target[1])

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.x0.seed(seed)

        self.state = self.x0.sample()
        self.step_count = 0
        return self.state, {}

    def Q(self, x):
        """Q(x) computes the spatial preference at x.
        If x.ndim == 2, each row of x is a new state
        """
        if x.ndim < 2:
            x = np.expand_dims(x, axis=0)

        c = self.max_height
        z = c * np.exp(-0.5 * self.slope * np.linalg.norm(x, axis=1) ** 2)
        return z.squeeze()

    def state2grid(self, state: np.array):
        state_dis = discretize_box_state(state, self.grid_size, self.observation_space)
        return state_dis
        # state_dis = np.floor((state - self.domain[0::2] - 0.5 * self.h) / self.h).astype(np.int8)
        # return np.minimum(self.grid_size - 1, np.maximum(0, state_dis))

    def grid2state(self, state_dis: np.array):
        # TODO: add noise so not always in same spacial location
        # state_dis = np.maximum(state_dis, np.array([0, 0], dtype=state_dis.dtype))
        # state_dis = np.minimum(state_dis, np.array(self.grid_size - 1, dtype=state_dis.dtype))
        # return np.minimum(self.h * state_dis + self.low + 0.5 * self.h, self.high)
        # return self.h * state_dis + self.domain[0::2] + 0.5 * self.h
        return self.h * state_dis + self.low + 0.5 * self.h

    def step(self, u):
        """step(u) applies action u to the state. u can be in {0,1,2,3}
            Outputs:
                self.state - new state
                r      - reward of current step
                done   - bool, True if episode ends because :
                          1) target was reached
                          2) we took too many steps
                          3) we ran out of the domain
                info   - empty. can hold additional variables for plotting
        """

        err_msg = "%r (%s) invalid" % (u, type(u))
        assert self.action_space.contains(u), err_msg

        # reward for current state (will be given upon step)
        r = -max(self.h) * self.Q(self.state).item()

        # discretize
        x, y = self.state2grid(self.state)
        x_old, y_old = deepcopy(x), deepcopy(y)
        if u == 0:
            # move left
            x -= 1
            r -= self.h[0]
        elif u == 1:
            # move right
            x += 1
            r -= self.h[0]
        elif u == 2:
            # move down
            y -= 1
            r -= self.h[1]
        else:  # u = 3
            # move up
            y += 1
            r -= self.h[1]

        # stay in domain
        x = np.clip(x, 0, self.grid_size[0] - 1)
        y = np.clip(y, 0, self.grid_size[1] - 1)

        # if don't move
        stay_still = (x == x_old and y == y_old)

        # if leave domain
        left_domain = (x == -1 or x == self.grid_size[0] or y == -1 or y == self.grid_size[1])

        # if reach target
        reached_target = (x == self.x_target_dis[0] and y == self.x_target_dis[1])

        self.step_count += 1
        max_steps = (self.step_count >= self.max_step_count)

        # done = bool(reached_target or left_domain or max_steps)
        done = bool(reached_target or left_domain)

        # update state
        self.state = self.grid2state(np.array([x, y])).astype(self.state.dtype)

        # r -= self.Q(self.state).item()

        # terminal cost
        if done:
            r += 0.0

        return self.state, r, done, False, {}

    def render(self, mode='human', close=False, plot=True):
        raise NotImplementedError


class TrajectoryEnvVisualizer:
    def __init__(self, traj: TrajectoryEnv):
        super(TrajectoryEnvVisualizer, self).__init__()
        self.traj = traj

        self.domain_ext = [traj.observation_space.low[0] - traj.h[0],
                           traj.observation_space.high[0] + traj.h[0],
                           traj.observation_space.low[1] - traj.h[1],
                           traj.observation_space.high[1] + traj.h[1]]

        self.domain = [traj.observation_space.low[0], traj.observation_space.high[0],
                       traj.observation_space.low[1], traj.observation_space.high[1]]

        self.grid_size = traj.grid_size

        x_grid, y_grid = np.meshgrid(
            np.linspace(self.domain[0] + 0.5 * traj.h[0], self.domain[1] - 0.5 * traj.h[0], self.grid_size[0]),
            np.linspace(self.domain[2] + 0.5 * traj.h[1], self.domain[3] - 0.5 * traj.h[1], self.grid_size[1]))

        self.x_grid = x_grid
        self.y_grid = y_grid

        self.real_world = traj.Q(np.concatenate((x_grid.reshape(-1, 1),
                                                 y_grid.reshape(-1, 1)), axis=1)).reshape(x_grid.shape)
        self.real_world_pad = np.pad(self.real_world, 1, 'constant', constant_values=self.real_world.max() + 1)
        self.x_start = traj.grid2state(traj.state2grid(deepcopy(traj.reset()[0])))
        self.x_target = traj.grid2state(traj.state2grid(deepcopy(traj.x_target)))

        cmap = plt.get_cmap('viridis')
        cmap.set_over('w')
        cmap.set_under('k')
        self.cmap = cmap
        self.cmap_grey =  plt.get_cmap('gist_gray')

    def plot_world(self, show_start=False, show_end=False, show_colorbar=False, show_legend=False, show_axis=False):
        vmax = self.real_world.max()

        plt.imshow(self.real_world_pad, origin='lower', extent=self.domain_ext, cmap=self.cmap, vmax=vmax)

        if show_start:
            plt.plot(self.x_start[0], self.x_start[1], 'go', markersize=10, markeredgewidth=1.2, label='start',
                     markeredgecolor='k')

        if show_end:
            plt.plot(self.x_target[0], self.x_target[1], 'r8', markersize=10, label='end', markeredgecolor='k')

        if show_colorbar:
            plt.colorbar(fraction=0.03, pad=0.01)

        if show_legend:
            plt.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.05))

        if show_axis is False:
            plt.axis('off')

    def get_values(self, values=None):
        raise NotImplementedError

    def get_greedy_actions(self, values=None):
        my_values = self.get_values(values=values)
        return my_values.argmax(axis=-1)

    def get_greedy_path(self, values=None):
        greedy_actions = self.get_greedy_actions(values=values)

        state = self.traj.grid2state(self.traj.state2grid(self.traj.reset()[0]))
        greedy_path = deepcopy(state).reshape(1, -1)
        done = False
        iter = 0
        while not done:
            state_dis = self.traj.state2grid(state)

            a = greedy_actions[tuple(state_dis)]
            state, r, done, info, _ = self.traj.step(a)

            tmp = deepcopy(state).reshape(1, -1)
            greedy_path = np.concatenate((greedy_path, tmp), axis=0)

            # # take too many steps
            if iter >= self.traj.max_step_count:
                done = True

            # # leave domain
            if not self.traj.observation_space.contains(state):
                done = True

            iter += 1

        return greedy_path

    def plot_greedy_policy(self, values=None, show_start=False, show_end=False, show_colorbar=False, show_legend=False, show_axis=False):
        greedy_actions = self.get_greedy_actions(values=values)

        self.plot_world(show_start=show_start, show_end=show_end, show_colorbar=show_colorbar)

        my_params = {'color': 'black', 'width': 0.0075, 'scale': 15}

        for i, u in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            idx = (greedy_actions.T == i)
            xx = self.x_grid[idx] - 0.1 * u[0] * self.traj.h[0]
            yy = self.y_grid[idx] - 0.1 * u[1] * self.traj.h[0]
            plt.quiver(xx, yy, u[0], u[1], **my_params)

    def plot_greedy_path(self, values=None, show_start=False, show_end=False, show_colorbar=False, show_legend=False, show_axis=False):

        greedy_path = self.get_greedy_path(values=values)

        self.plot_world(show_start=show_start, show_end=show_end, show_colorbar=show_colorbar, show_legend=show_legend, show_axis=show_axis)

        plt.plot(greedy_path[:, 0], greedy_path[:, 1], '-k', linewidth=3.0)

    def max_value_plot_3d(self, values=None, show_start=False, show_end=False):
        values = self.get_values(values=values)

        values_max = values.max(axis=-1)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(self.x_grid, self.y_grid, values_max, cmap=self.cmap)

        if show_start:
            ax.scatter3D(self.x_start[0], self.x_start[1], values_max.min(), color='g')

        if show_end:
            ax.scatter3D(self.x_target[0], self.x_target[1], values_max.min(), color='r')

    def state_action_value_plot(self, action, show_colorbar=False, show_start=False, show_end=False, show_legend=False, show_axis=False):
        my_values = self.get_values()
        vmax = my_values.max()
        plt.imshow(np.pad(my_values[:, :, action], 1, 'constant', constant_values=vmax + 1).T,
                   origin='lower', extent=self.domain_ext, cmap=self.cmap)

        if show_start:
            plt.plot(self.x_start[0], self.x_start[1], 'go', markersize=10, markeredgewidth=1.2,
                     label='start', markeredgecolor='k')

        if show_end:
            plt.plot(self.x_target[0], self.x_target[1], 'r8', markersize=10, label='end', markeredgecolor='k')

        if show_colorbar:
            plt.colorbar(fraction=0.0421, pad=0.01)

        if show_legend:
            plt.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.05))

        if show_axis is False:
            plt.axis('off')

    def state_action_value_plot_3d(self, action):
        # my_params = {'color': 'black', 'scale': 15}

        dx = 0
        if action == 0:
            dx = -1
        if action == 1:
            dx = 1

        dy = 0
        if action == 2:
            dy = -1
        if action == 3:
            dy = 1

        my_values = self.get_values()
        vmax = my_values.max()
        vmin = my_values.min()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(self.x_grid, self.y_grid, my_values[:, :, action], cmap=self.cmap)
        plt.xlabel('x')
        plt.ylabel('y')
        ax.quiver(self.x_grid, self.y_grid,
                  vmin - 1 + 0 * self.x_grid, 0 * self.x_grid + dx,
                  0 * self.x_grid + dy, 0 * self.x_grid,
                  length=0.1, color='k')


class TrajectoryEnvVisualizersFunctionApproximation(TrajectoryEnvVisualizer):

    def __init__(self, traj: TrajectoryEnv, model):
        super(TrajectoryEnvVisualizersFunctionApproximation, self).__init__(traj)
        self.model = model

    def get_values(self, values=None):

        if values is None:
            xy = self.traj.grid2state(self.traj.state2grid(np.concatenate((self.x_grid.reshape(-1, 1),
                                                                           self.y_grid.reshape(-1, 1)), axis=-1)))
            x_coor = self.x_grid[0]
            y_coor = self.y_grid[:, 0]

            values = np.zeros([self.x_grid.shape[1], self.y_grid.shape[0], 4])
            for i in range(self.x_grid.shape[1]):
                for j in range(self.y_grid.shape[0]):
                    values[i, j] = self.model(np.array([x_coor[i], y_coor[j]])).detach().numpy()

        return values


class TrajectoryEnvVisualizersTabular(TrajectoryEnvVisualizer):

    def __init__(self, traj: TrajectoryEnv, Q):
        super(TrajectoryEnvVisualizersTabular, self).__init__(traj)
        self.Q = Q

    def get_values(self, values=None):
        if values is None:
            values = self.Q
        return values


if __name__ == "__main__":
    from rl_trajectory.algorithms import SemiGradientSARSALinearModel
    from rl_trajectory.featurizers import PolynomialFeaturizer
    from rl_trajectory.models import LinearModel

    traj = TrajectoryEnv(grid_size=np.array((16, 16)))

    featurizer = PolynomialFeaturizer(order=2)

    num_actions = traj.action_space.n
    num_state = traj.observation_space.shape[0]
    num_features = featurizer.num_features(num_state)
    model = LinearModel(num_features, num_actions)

    opt = SemiGradientSARSALinearModel(model, featurizer)

    visualizer = TrajectoryEnvVisualizersFunctionApproximation(traj, opt, featurizer)

    visualizer.plot_world(show_colorbar=True, show_start=True, show_end=True, show_legend=True, show_axis=True)
    plt.show()

    visualizer.plot_greedy_path(show_colorbar=True, show_start=True, show_end=True)
    plt.show()

    visualizer.plot_greedy_policy(show_colorbar=True, show_start=True, show_end=True)
    plt.show()

