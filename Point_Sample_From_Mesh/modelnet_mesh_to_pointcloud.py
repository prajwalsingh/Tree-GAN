from meshtopc import genereate_point_cloud
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
	mesh_train_path = natsorted(glob('data/ModelNet10/*/train/*.off'))
	mesh_test_path  = natsorted(glob('data/ModelNet10/*/test/*.off'))
	N = 2048 # Number of points in point cloud

	# pc_train_path    = os.path.dirname(mesh_train_path[0]).split(os.path.sep)
	# pc_train_path[1] = 'PC'+pc_train_path[1]
	# os.path.sep.join(pc_train_path)

	print('Creating Train Point Cloud')
	for train_path in tqdm(mesh_train_path):
		P = genereate_point_cloud(train_path, N)
		file_name  = os.path.splitext(os.path.basename(train_path))[0] + '_pc.npy'
		pc_path    = os.path.dirname(train_path).split(os.path.sep)
		pc_path[1] = 'PC_'+pc_path[1]+'_2048'
		pc_path    = os.path.sep.join(pc_path)
		if not os.path.isdir(pc_path):
			os.makedirs(pc_path)
		pc_path    = os.path.join(pc_path, file_name)
		np.save(pc_path, P)

	print('Creating Test Point Cloud')
	for test_path in tqdm(mesh_test_path):
		P = genereate_point_cloud(test_path, N)
		file_name  = os.path.splitext(os.path.basename(test_path))[0] + '_pc.npy'
		pc_path    = os.path.dirname(test_path).split(os.path.sep)
		pc_path[1] = 'PC_'+pc_path[1]+'_2048'
		pc_path    = os.path.sep.join(pc_path)
		if not os.path.isdir(pc_path):
			os.makedirs(pc_path)
		pc_path    = os.path.join(pc_path, file_name)
		np.save(pc_path, P)