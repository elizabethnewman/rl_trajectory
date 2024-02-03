import os
import datetime
import torch
from rl_trajectory.utils import seed_everything, get_argument_parser, get_logger, makedirs
import rl_trajectory.envs as env
import rl_trajectory.algorithms as alg
import pandas as pd
import numpy as np
import gym


# get arguments
parser = get_argument_parser()
args = parser.parse_args()

# seed for reproducibility
seed_everything(args.seed)

# setup logger
file_name = os.path.splitext(os.path.basename(__file__))[0]
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# create folder to store log file (we may want to choose a different naming convention)
savedir_name = os.path.join(args.savelog, start_time)
makedirs(savedir_name)

logger = get_logger(logpath=os.path.join(savedir_name, 'logs'), filepath=os.path.abspath(__file__))
logger.info("start time: {:}".format(start_time))

logger.info("args: {:}".format(args))

# set up environment
traj = env.TrajectoryEnv(grid_size=np.array((args.grid_size, args.grid_size)), max_step_count=args.max_steps)
traj.x0 = gym.spaces.Box(low=np.array([-1, -1], dtype=np.float32),
                         high=np.array([1, 1], dtype=np.float32),
                         dtype=np.float32)

# setup optimizer
opt = alg.TabularSARSA(max_steps=args.max_steps, num_episodes=args.num_episodes,
                       max_eps=args.max_eps, min_eps=args.min_eps, alpha=args.alpha)

Q = np.zeros(tuple(traj.grid_size) + (traj.action_space.n, ), dtype=np.float32)
results = opt.train(traj, Q, verbose=args.verbose, log_interval=args.log_interval)

# Save iteration history as csv
pd.DataFrame.to_csv(pd.DataFrame(results['history']['value'], columns=results['history']['header']),
                    os.path.join(savedir_name, 'history.csv'))

# Save the whole model (final model, best model)
np.save(os.path.join(savedir_name, 'final_model.npy'), Q)
torch.save(results, os.path.join(savedir_name, 'results.pth'))

#%% visualize

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
plt.savefig(os.path.join(savedir_name, 'total_reward.png'),  bbox_inches='tight')

# reset to usual starting plot
s0 = [-1 + traj.h[0], -1 + traj.h[1]]
traj.x0 = gym.spaces.Box(np.array(s0, dtype=np.float32), np.array(s0, dtype=np.float32), dtype=np.float32)

visualizer = env.TrajectoryEnvVisualizersTabular(traj, Q)

visualizer.plot_greedy_path(show_colorbar=True, show_start=True, show_end=True)
plt.savefig(os.path.join(savedir_name, 'greedy_path.png'),  bbox_inches='tight')
plt.close()

visualizer.plot_greedy_policy(show_colorbar=True, show_start=True, show_end=True)
plt.savefig(os.path.join(savedir_name, 'greedy_policy.png'), bbox_inches='tight')

values = visualizer.get_values()

values_max = values.max(axis=-1)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(visualizer.x_grid, visualizer.y_grid, values_max, cmap=visualizer.cmap)
ax.scatter3D(visualizer.x_start[0], visualizer.x_start[1], values_max.min(), color='g')
ax.scatter3D(visualizer.x_target[0], visualizer.x_target[1], values_max.min(), color='r')
plt.savefig(os.path.join(savedir_name, 'value_function.png'),  bbox_inches='tight')
