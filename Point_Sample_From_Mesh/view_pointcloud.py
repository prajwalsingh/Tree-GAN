from mpl_toolkits import mplot3d
import  matplotlib.pyplot as plt
from glob import glob
from natsort import natsorted
import numpy as np
import os

if __name__ == '__main__':
	pc_train_path = natsorted(glob('data/PC_ModelNet10_2048/*/train/*.npy'))
	pc_test_path  = natsorted(glob('data/PC_ModelNet10_2048/*/test/*.npy'))

	for train_path in pc_train_path[::300]:
		P = np.load(train_path)
		ax = plt.axes(projection='3d')
		ax.scatter3D(P[:, 0], P[:, 1], P[:, 2])
		plt.title(train_path.split(os.path.sep)[2])
		# plt.show()
		plt.pause(0.5)
		plt.clf()

	for test_path in pc_test_path[::300]:
		P = np.load(test_path)
		ax = plt.axes(projection='3d')
		ax.scatter3D(P[:, 0], P[:, 1], P[:, 2])
		plt.title(test_path.split(os.path.sep)[2])
		# plt.show()
		plt.pause(0.5)
		plt.clf()