import rl_trajectory.envs as env
import rl_trajectory.algorithms as alg
import rl_trajectory.featurizers
import rl_trajectory.models as models
import gym
import torch
import numpy as np
from rl_trajectory.utils import set_seed
import argparse
from rl_trajectory.utils import insert_data, get_argument_parser, seed_everything

# get arguments
parser = get_argument_parser()
args = parser.parse_args()


#%%

traj = env.TrajectoryEnv(grid_size=np.array((16, 16)), max_step_count=args.max_steps)

# start from anywhere
traj.x0 = gym.spaces.Box(low=np.array([-1, -1], dtype=np.float32),
                         high=np.array([1, 1], dtype=np.float32),
                         dtype=np.float32)

# start from lower left
# low = np.array([-1, -1], dtype=np.float32)
# start_domain_step = 15
# traj.x0 = gym.spaces.Box(low=low,
#                          high=low + start_domain_step * traj.h,
#                          dtype=np.float32)

featurizer = rl_trajectory.featurizers.PolynomialFeaturizer(order=2)

num_actions = traj.action_space.n
num_state = traj.observation_space.shape[0]
num_features = featurizer.num_features(num_state)
model = models.LinearModel(featurizer, num_state, num_actions)

print('initial weights =', model.w)
print('num_features =', num_features)


opt = alg.LinearSemiGradientSARSA(**vars(args))


#%%
seed_everything(123, traj)
Q, S, info = opt.train(traj, model, verbose=True, log_interval=1)

#%% convergence plots
import matplotlib.pyplot as plt

info = opt.history
info['value'] = np.array(info['value'])
idx1 = info['header'].index('episode')
idx2 = info['header'].index('total_reward')

plt.figure()
plt.plot(info['value'][:, idx1], info['value'][:, idx2], linewidth=2)
plt.xlabel('episode')
plt.ylabel('return')
plt.title('total reward')
plt.show()

#%%
from copy import deepcopy
# reset to usual starting plot
s0_dis = [1, 1]
s0 = [-1 + s0_dis[0] * traj.h[0], -1 + s0_dis[1] * traj.h[1]]
traj.x0 = gym.spaces.Box(np.array(s0, dtype=np.float32), np.array(s0, dtype=np.float32), dtype=np.float32)

model_best = deepcopy(model)
insert_data(model_best, opt.best['w'])
visualizer = env.TrajectoryEnvVisualizersFunctionApproximation(traj, model)

#%%
# visualizer.plot_world()
# plt.show()

visualizer.plot_greedy_path(show_colorbar=True, show_start=True, show_end=True)
plt.show()

visualizer.plot_greedy_policy(show_colorbar=True, show_start=True, show_end=True)
plt.show()

# plt.figure()
# for i, name in enumerate(['left', 'right', 'down', 'up']):
#     plt.subplot(2, 2, i + 1)
#     visualizer.state_action_value_plot(i, show_colorbar=True, show_start=True, show_end=True)
#     plt.title(name)
#
# plt.show()

#%%
values = visualizer.get_values()

values_max = values.max(axis=-1)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(visualizer.x_grid, visualizer.y_grid, values_max, cmap=visualizer.cmap)
ax.scatter3D(visualizer.x_start[0], visualizer.x_start[1], values_max.min(), color='g')
ax.scatter3D(visualizer.x_target[0], visualizer.x_target[1], values_max.min(), color='r')
plt.show()
