import numpy as np

#####################################################################
#						 Neural Net Class							#
#####################################################################

class Neural_Net():
	def __init__(self, layers=2, nodes_per_layer, activation_fn='relu', lr=0.001, gamma=0.5):
		"""
		Returns neural net object

		Hidden layers including prediction are indexed from 0.
		"""
		self.hidden_layers = layers-2
		self.activation_fn = activation_fn

		assert isinstance(nodes_per_layer, list)

		self.weights = dict()
		self.bias = dict()
		nodes_per_hidden_layer = nodes_per_layer[1:]

		for i in range(layers-1):
			self.weights[i] = 0.01 * np.random.normal(0, 1, nodes_per_hidden_layer[i])
			self.bias[i] = np.zeros(nodes_per_hidden_layer[i])


	def fit(self, train_data, train_labels, batch_size, epochs):
		"""
		Train weights and biases!
		Assume all preprocessing has been done.
		"""
		raise NotImplementedError

		for epoch in range(epochs):


	def predict(self, test_data):
		"""
		Return class probabilites for test data.
		"""
		raise NotImplementedError

	def score(self, test_data, test_labels):
		"""
		Return prediction accuracy.
		"""		
		raise NotImplementedError

	def relu(self, x):
		"""
		Return ReLU activation
		"""
		raise NotImplementedError

	def sigmoid(self, x):
		"""
		Return sigmoid activation
		"""
		raise NotImplementedError

	def linear(self, x):
		"""
		Return Linear activation
		"""
		raise NotImplementedError

	def tanh(self, x):
		"""
		Return Tanh activation
		"""
		raise NotImplementedError

	def softmax(self, x):
		"""
		Return Softmax activation
		"""
		raise NotImplementedError

	def d_relu(self, x):
		"""
		Return derivative of ReLU
		"""
		raise NotImplementedError

	def d_sigmoid(self, x):
		"""
		Return derivative of Sigmoid
		"""
		raise NotImplementedError

	def d_linear(self, x):
		"""
		Return derivative of Linear
		"""
		raise NotImplementedError

	def d_tanh(self, x):
		"""
		Return derivative of Tanh
		"""
		raise NotImplementedError

	def d_softmax(self, x):
		"""
		Return derivative of Softmax
		"""
		raise NotImplementedError

	def lr_decay(self, current_lr):
		"""
		Decay current learning rate.
		"""
		self.lr = self.lr * self.gamma

	def has_loss_converged(self, curr_loss, prev_loss):
		"""
		Decay lr if True, check per epoch?
		"""
		if abs(curr_loss - prev_loss)<=1e-4:
			return True
		return False

	def get_CE_loss(self, soft_fc_output, truth_dist):
		"""
		Return cross entropy loss. Assuming batch to be 
		something like mxn shape, 
		"""
		raise NotImplementedError

#####################################################################
#						  Auxillary Methods							#
#####################################################################
def normalize(data):
	"""
	Return feature(pixel) normalization by mean
	and standard deviation across examples and 
	a feature wise dictionary with (mean, std) for
	later.
	"""
	normalize_dict = dict()

	_, feat_len = data.shape
	assert feat_len==784
	
	temp_data = np.copy(data)
	for i in range(feat_len):
		col_mean = np.mean(data[:, i])
		col_std = np.std(data[:, i])
		print(data[:, i], data[:, i].shape, col_mean, col_std)
		input('continue?')

		temp_data[:, i] = (data[:, i] - col_mean)/col_std
		normalize_dict[i] = (col_mean, col_std)

	return temp_data, normalize_dict


def get_mnist(path='./MNIST'):
	"""
	Return MNIST data.
	"""
	from mnist import MNIST
	data = MNIST(path)
	train_images, train_labels = data.load_training()
	test_images, test_labels = data.load_testing()

	train_images = np.reshape(np.asarray(train_images), (60000, 784))
	train_labels = np.reshape(np.asarray(train_labels), (60000,))
	test_images = np.reshape(np.asarray(test_images), (10000, 784))
	test_labels = np.reshape(np.asarray(test_labels), (10000,))

	return train_images, train_labels, test_images, test_labels

#####################################################################
#							  Questions								#
#####################################################################

#####################################################################
#								Main								#
#####################################################################
