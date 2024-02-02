import rl_trajectory.envs as env
import rl_trajectory.algorithms as alg
import numpy as np
import gym
from rl_trajectory.utils import seed_everything

max_steps = 1000

traj = env.TrajectoryEnv(grid_size=np.array((16, 16)), max_step_count=max_steps)
traj.x0 = gym.spaces.Box(low=np.array([-1, -1], dtype=np.float32),
                         high=np.array([1, 1], dtype=np.float32),
                         dtype=np.float32)

seed_everything(42, traj)

opt = alg.TabularSARSA(max_steps=max_steps, num_episodes=3000, penalty=0, max_eps=1e0, min_eps=1e-5, alpha=0.5)

Q = np.zeros(tuple(traj.grid_size) + (traj.action_space.n, ), dtype=np.float32)
results = opt.train(traj, Q, verbose=True, log_interval=10)

#%%
import matplotlib.pyplot as plt


info = results['history']
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
# reset to usual starting plot
s0 = [-1 + traj.h[0], -1 + traj.h[1]]
traj.x0 = gym.spaces.Box(np.array(s0, dtype=np.float32), np.array(s0, dtype=np.float32), dtype=np.float32)

visualizer = env.TrajectoryEnvVisualizersTabular(traj, Q)

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
