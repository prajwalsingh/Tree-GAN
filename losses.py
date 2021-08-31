import tensorflow as tf

def gen_loss(D_fake):
	return -tf.reduce_mean(D_fake)

def critic_loss_gp(D_real, D_fake, Y, Y_cap, model, batch_size):
	dloss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
	lam   = 10
	eps   = tf.random.uniform(shape=[batch_size, 1, 1], minval=0, maxval=1)
	x_cap = eps * Y + (1-eps) * Y_cap
	with tf.GradientTape() as gptape:
		gptape.watch(x_cap)
		out = model.critic(x_cap, training=True)
	grad = gptape.gradient(out, x_cap)[0]
	grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[0, 1]))
	grad_pen  = tf.reduce_mean((grad_norm - 1.0)**2)
	dloss = dloss + lam * grad_pen
	return dloss