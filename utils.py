from glob import glob
from natsort import natsorted
import os
import numpy as np
import tensorflow as tf
import functools
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

batch_size = 16

def fetch_data(path, cat_dict):
	path = path.numpy().decode('utf-8')
	X    = np.load(path)
	Y    = cat_dict[path.split(os.path.sep)[1]]
	return X, Y

def load_batch(path, batch_size=16):
	return tf.data.Dataset.from_tensor_slices((path)).shuffle(len(path)).batch(batch_size, drop_remainder=True)

def load_data(path, cat_dict):
	X, Y = list(zip(*map(functools.partial(fetch_data, cat_dict=cat_dict), path)))
	X    = tf.convert_to_tensor(X, dtype='float32')
	Y    = tf.convert_to_tensor(Y, dtype='int32')
	return X, Y

def view_data(X, Y, cat_dict, rev_cat_dict):
	if not os.path.isdir('save_fig'):
		os.makedirs('save_fig')

	for idx, (shape, cat) in enumerate(zip(X, Y)):
		shape = shape.numpy()
		cat   = cat.numpy()
		ax    = plt.axes(projection='3d')
		ax.scatter3D(shape[:, 0], shape[:, 1], shape[:, 2])
		plt.title('Category: {0}'.format(rev_cat_dict[cat]))
		plt.savefig('save_fig/{0}.png'.format(idx))

def view_results(model, latent_dim, batch_size, categ, fileno):
	if not os.path.isdir('results'):
		os.makedirs('results/train/')
		os.makedirs('results/val/')
	idx   = np.random.randint(0, batch_size-1)
	tree  = [tf.random.uniform(shape=[batch_size, 1, latent_dim])]
	X     = model.gen(tree, training=False)[idx]
	ax    = plt.axes(projection='3d')
	ax.scatter3D(X[:, 0], X[:, 1], X[:, 2])
	plt.savefig('results/{0}/{1}.png'.format(categ, fileno))