import tensorflow as tf
import os
from natsort import natsorted
from glob import glob
from utils import load_data, load_batch, view_results
from model import TreeGAN, train_step, val_step
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

if __name__ == '__main__':
	batch_size   = 16
	latent_dim   = 96
	K            = 10 # Support
	train_path   = natsorted(glob('data/*/train/*.npy'))
	val_path     = natsorted(glob('data/*/test/*.npy'))
	cat_dict     = {} # Category dictionary from class to idx
	rev_cat_dict = {} # Category dictionary from idx to class
	for idx, cat in enumerate(natsorted(glob('data/*'))):
		cat_dict[cat.split(os.path.sep)[1]] = idx
		rev_cat_dict[idx] = cat.split(os.path.sep)[1]

	train_batch = load_batch(train_path, batch_size=batch_size)
	val_batch   = load_batch(val_path, batch_size=batch_size)
	model       = TreeGAN(K, latent_dim, batch_size)
	gen_opt     = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5, beta_2=0.9)
	critic_opt  = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5, beta_2=0.9)
	ckpt        = tf.train.Checkpoint(step=tf.Variable(1), model=model, gopt=gen_opt, copt=critic_opt)
	manager     = tf.train.CheckpointManager(ckpt, directory='treegan_ckpt', max_to_keep=10)
	ckpt.restore(manager.latest_checkpoint)
	EPOCHS      = 5000
	START       = int(ckpt.step) // len(train_batch) + 1
	save_freq   = 100
	tvis_freq   = 100
	vvis_freq   = 20
	if manager.latest_checkpoint:
		print('Restored from last checkpoint, epoch : {0}'.format(START))

	for epoch in range(START, EPOCHS):
		train_gloss = tf.keras.metrics.Mean()
		train_closs = tf.keras.metrics.Mean()
		val_gloss   = tf.keras.metrics.Mean()

		for idx, path in enumerate(tqdm(train_batch), start=1):
			X, Y = load_data(path, cat_dict)
			gloss, closs = train_step(model, gen_opt, critic_opt, X, latent_dim, batch_size)
			train_gloss.update_state(gloss)
			train_closs.update_state(closs)
			ckpt.step.assign_add(1)
			if (idx%save_freq) == 0:
				manager.save()
			if (idx%tvis_freq) == 0:
				view_results(model, latent_dim, batch_size, 'train', int(ckpt.step))
			print('Train_GLoss: {0}\tTrain_CLoss: {1}'.format(gloss, closs))


		for idx, path in enumerate(tqdm(val_batch), start=1):
			X, Y = load_data(path, cat_dict)
			gloss = val_step(model, X, latent_dim, batch_size)
			val_gloss.update_state(gloss)
			if (idx%vvis_freq) == 0:
				view_results(model, latent_dim, batch_size, 'val', int(ckpt.step)+idx)
			print('Val_GLoss: {0}'.format(gloss))

		with open('log.txt', 'a') as file:
			file.write('Epoch: {0}\tTrain_GLoss: {1}\tTrain_CLoss: {2}\tVal_GLoss: {3}\n'.format(epoch, train_gloss.result(), train_closs.result(), val_gloss.result()))

		print('Epoch: {0}\tTrain_GLoss: {1}\tTrain_CLoss: {2}\tVal_GLoss: {3}'.format(epoch, train_gloss.result(), train_closs.result(), val_gloss.result()))