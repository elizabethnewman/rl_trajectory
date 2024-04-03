# https://github.com/lbarazza/Tile-Coding
import numpy as np
from rl_trajectory.featurizers import Featurizer
from torch import Tensor
from typing import Union


class TilingFeaturizer(Featurizer):

	def __init__(self, grid_size: np.ndarray, low, high, n_tilings, offset_type='symmertical'):
		"""
		grid_size   : number of rectangular tiles
		low         : lower bounds of continuous domain
		high        : upper bounds of continuous domain
		n_tilings   : number of tilings (number of offsets)
		offset      : type of offset (uniform, for now)
		"""
		super(TilingFeaturizer, self).__init__()
		self.grid_size = np.array(grid_size, dtype=np.int8)
		self.low = low
		self.high = high
		self.n_tilings = n_tilings
		self.offset_type = offset_type

		self.n_feat = np.prod(grid_size) * n_tilings

		# get low and high endpoints for each tiling
		if offset_type == 'asymmetric':
			raise NotImplementedError
		else:
			n_box = (grid_size - 1) * n_tilings + 1  # number of gridpoints in refined grid
			w = (high - low) / n_box  # width of each refined grid box
			h2 = (high - w - low) / (grid_size - 1)
			low_tilings = low.reshape(1, -1) - ((h2 - w.reshape(1, -1)) / (n_tilings - 1)).reshape(1, -1) * np.arange(0, n_tilings).reshape(-1, 1)
			high_tilings = high.reshape(1, -1) + ((h2 - w.reshape(1, -1)) / (n_tilings - 1)).reshape(1, -1) * np.arange(n_tilings - 1, -1, -1).reshape(-1, 1)
			# h_tilings = (high_tilings - low_tilings) / grid_size

		self.low_tilings = low_tilings
		self.high_tilings = high_tilings

	def _discretize_state(self, state):
		# find step size
		h = (self.high_tilings - self.low_tilings) / self.grid_size.reshape(1, -1)

		# discretize
		state_dis = np.floor((state.reshape(1, -1) - self.low_tilings) / h).astype(np.int8)

		# make sure we do not leave grid
		state_dis = np.minimum(self.grid_size - 1, np.maximum(0, state_dis))

		return state_dis

	def num_features(self, *args) -> int:
		return self.n_feat

	def forward(self, state: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
		# TODO: improve efficiency - only need to store indices
		# TODO: create coarse coding featurizer - when we move to models instead of linear features, we can flag this and treat it differently
		state_dis = self._discretize_state(state)
		state_dis = np.hsplit(state_dis, np.arange(1, state_dis.shape[1]))
		state_feat = np.zeros(self.n_feat).reshape(tuple(self.grid_size) + (self.n_tilings,))
		state_feat[tuple(state_dis) + (np.arange(self.n_tilings).reshape(-1, 1),)] = 1.0
		return state_feat.reshape(-1)


if __name__ == "__main__":
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib import cm

	grid_size = np.array([3, 5])
	low = np.array([-1, -1])
	high = np.array([1, 1])
	n_tilings = 3
	tmp = TilingFeaturizer(grid_size, low, high, n_tilings)

	s = np.array([-0.4, 0.6])
	s_dis = tmp._discretize_state(s)
	print(s_dis)

	plt.figure()

	h = (high - low) / grid_size
	x, y = np.meshgrid(np.linspace(low[0] + h[0] / 2, high[0] - h[0] / 2, grid_size[0]),
					   np.linspace(low[1] + h[1] / 2, high[1] - h[1] / 2, grid_size[1]))
	plt.pcolor(x, y, np.ones(x.shape), cmap=plt.cm.gray)

	h = (tmp.high_tilings - tmp.low_tilings) / grid_size.reshape(1, -1)

	for i, mycolor in enumerate(['r', 'b', 'y', 'c']):
		if i < n_tilings:
			x, y = np.meshgrid(np.linspace(tmp.low_tilings[i][0] + h[i][0] / 2, tmp.high_tilings[i][0] - h[i][0] / 2, grid_size[0]),
							   np.linspace(tmp.low_tilings[i][1] + h[i][1] / 2, tmp.high_tilings[i][1] - h[i][1] / 2, grid_size[1]))
			plt.pcolor(x, y, np.ones(x.shape), alpha=0.25, edgecolor=mycolor, linewidth=2)
	plt.axis('equal')
	plt.plot(s[0], s[1], 'wx', linewidth=2)
	plt.show()





