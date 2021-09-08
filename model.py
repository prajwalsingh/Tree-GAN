import tensorflow as tf
from tensorflow.keras import Model, layers
from losses import gen_loss, critic_loss_gp

class TreeGCN(Model):
	def __init__(self, depth, child_node, degrees, filters, K=10, batch_size=8, upsample=True, activation=True):
		super(TreeGCN, self).__init__()
		self.depth         = depth
		self.batch_size    = batch_size
		self.child_node    = child_node
		self.input_feat    = filters[depth]
		self.out_feat      = filters[depth+1]
		self.degree        = degrees[depth]
		self.upsample      = upsample
		self.activation    = activation
		self.F_K           = [layers.Dense(units=K * self.input_feat), layers.Dense(units=self.out_feat)] # Loop
		self.W_A           = [layers.Dense(units=self.out_feat) for i in range(depth+1)] # Ancestor weights
		self.b             = tf.Variable(tf.initializers.GlorotUniform()(shape=[1, self.degree, self.out_feat]), name='Bias') # Xavier Initalization
		self.W_upsample    = tf.Variable(tf.initializers.GlorotUniform()(shape=[child_node, self.input_feat, self.degree*self.input_feat]), name='WeightMat') # Xavier Initalization
		self.act           = layers.LeakyReLU(alpha=0.2)

	def call(self, tree):
		Gath = 0 # Use for information gathering
		# Step 1: Accumulate information from ancestor
		for layer_no in range(self.depth+1):
			anc_node   = tree[layer_no].shape[1]
			repeat_num = self.child_node // anc_node
			Q          = self.W_A[layer_no](tree[layer_no]) # Ancestors
			# print(layer_no, anc_node, self.child_node, repeat_num, tree[layer_no].shape, Q.shape, end=' ')
			Gath       = Gath + tf.reshape(tf.tile(Q, [1, 1, repeat_num]), [self.batch_size, -1, self.out_feat]) # Gathering all the information
			# print(Gath.shape)
		# Step 2: Upsample the tree and Pass it through loop
		next_level = 0 # Next level for tree
		if self.upsample:
			next_level = tf.expand_dims(tree[-1], axis=2) @ self.W_upsample
			next_level = self.act(next_level)
			next_level = tf.reshape(next_level, [self.batch_size, self.child_node*self.degree, self.input_feat])
			next_level = self.F_K[1]( self.F_K[0]( next_level ) )
			next_level = next_level + tf.reshape(tf.tile(Gath, [1, 1, self.degree]), [self.batch_size, -1, self.out_feat])
		else:
			next_level = self.F_K[1]( self.F_K[0]( tree[-1] ) )
			next_level = next_level + Gath
		# Step 3: Add bias and Pass it through Activation function
		if self.activation:
			next_level = self.act(next_level + tf.tile(self.b, [1, self.child_node, 1]))
		tree.append(next_level)
		return tree


class Generator(Model):
	def __init__(self, K=10, latent_dim=96, batch_size=8):
		super(Generator, self).__init__()
		filters         = [latent_dim, 256, 256, 256, 128, 128, 128,  3]
		degrees         = [         1,   2,   2,   2,   2,   2,  64]
		self.depth      = len(filters) - 1
		self.tree_layer = []
		child_node      = 1
		for layer_no in range(self.depth):
			if layer_no == self.depth-1:
				self.tree_layer.append(TreeGCN(layer_no, child_node, degrees, filters, K=K, batch_size=batch_size, upsample=True, activation=False))
			else:
				self.tree_layer.append(TreeGCN(layer_no, child_node, degrees, filters, K=K, batch_size=batch_size, upsample=True, activation=True))
			child_node  = child_node * degrees[layer_no]

	def call(self, tree):
		for layer_no in range(self.depth):
			tree = self.tree_layer[layer_no]( tree )
		return tree[-1]


class Critic(Model):
	def __init__(self):
		super(Critic, self).__init__()
		filters          = [3, 64,128,256,512,1024]
		units            = [128, 64]
		self.cnn_depth   = len(filters)
		self.fc_depth    = len(units)
		self.cnn_conv    = [layers.Conv1D(filters=filters[i], kernel_size=1, strides=1, padding='same') for i in range(self.cnn_depth)]
		self.cnn_act     = [layers.LeakyReLU(alpha=0.2) for _ in range(self.cnn_depth)]
		self.global_pool = layers.GlobalMaxPooling1D()
		self.flat        = layers.Flatten()
		self.fc          = [layers.Dense(units=units[i]) for i in range(self.fc_depth)]
		self.fc_act      = [layers.LeakyReLU(alpha=0.2) for _ in range(self.fc_depth)]
		self.fc_final    = layers.Dense(units=1)

	def call(self, x):
		for layer_no in range(self.cnn_depth):
			x = self.cnn_act[layer_no]( self.cnn_conv[layer_no]( x ) )
		x = self.global_pool(x)
		x = self.flat(x)
		for layer_no in range(self.fc_depth):
			x = self.fc_act[layer_no]( self.fc[layer_no]( x ) )
		x = self.fc_final(x)
		return x

class TreeGAN(Model):
	def __init__(self, K=10, latent_dim=96, batch_size=8):
		super(TreeGAN, self).__init__()
		self.gen    = Generator(K=K, latent_dim=latent_dim, batch_size=batch_size)
		self.critic = Critic()

def train_step(model, gen_opt, critic_opt, X, latent_dim, batch_size):
	critic_train = 5
	for _ in range(critic_train):
		with tf.GradientTape() as ctape:
			tree       = [tf.random.uniform(shape=[batch_size, 1, latent_dim])]
			fake_point = model.gen(tree, training=False)
			D_real     = model.critic(X, training=True)
			D_fake     = model.critic(fake_point, training=True)
			closs      = critic_loss_gp(D_real, D_fake, X, fake_point, model, batch_size)
		variables = model.critic.trainable_variables
		gradients = ctape.gradient(closs, variables)
		critic_opt.apply_gradients(zip(gradients, variables))

	with tf.GradientTape() as gtape:
		tree       = [tf.random.uniform(shape=[batch_size, 1, latent_dim])]
		fake_point = model.gen(tree, training=True)
		D_fake     = model.critic(fake_point, training=False)
		gloss      = gen_loss(D_fake)
	variables = model.gen.trainable_variables
	gradients = gtape.gradient(gloss, variables)
	gen_opt.apply_gradients(zip(gradients, variables))

	return gloss, closs

def val_step(model, X, latent_dim, batch_size):
	tree       = [tf.random.uniform(shape=[batch_size, 1, latent_dim])]
	fake_point = model.gen(tree, training=False)
	D_fake     = model.critic(fake_point, training=False)
	gloss      = gen_loss(D_fake)
	return gloss
