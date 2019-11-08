import numpy as np

#####################################################################
#						 Neural Net Class							#
#####################################################################

class Neural_Net():
	def __init__(self, layers=2, nodes_per_layer=[], activation_fn='relu', lr=0.001, gamma=0.5):
		"""
		Returns neural net object

		Hidden layers including prediction are indexed from 0.
		"""
		self.layers = layers
		self.hidden_layers = layers-2
		self.activation_fn = get_activation_fn(activation_fn)
		self.d_activation_fn = get_d_activation_fn(activation_fn)
		self.batch_size = 1
		assert isinstance(nodes_per_layer, list)

		self.weights = dict()
		self.bias = dict()
		# nodes_per_hidden_layer = nodes_per_layer[1:]

		for i in range(self.layers-1):
			self.weights[i] = 0.01 * np.random.normal(0, 1, (nodes_per_layer[i+1], nodes_per_layer[i]))
			self.bias[i] = np.zeros(nodes_per_layer[i+1])


	def fit(self, train_data, train_labels, batch_size, epochs):
		"""
		Train weights and biases!
		Assume all preprocessing has been done.
		"""
		raise NotImplementedError

		self.batch_size = batch_size
		sample_count, feat_count = list(train_data.shape)
		train_len = sample_count//self.batch_size
		prev_epoch_loss = float('inf')
		epoch_error = np.zeros(shape=(epochs,))

		for epoch in range(epochs):
			epoch_loss = 0
			shuffled_permutation = np.random.permutation(sample_count)
			shuffled_data = np.copy(train_data[shuffled_permutation])
			shuffled_labels = np.copy(train_labels[shuffled_permutation])

			for batch in train_len:
				train_batch = shuffled_data[batch*self.batch_size:(batch+1)*self.batch_size]
				label_batch = shuffled_labels[batch*self.batch_size:(batch+1)*self.batch_size]

				fc_output = self.forward(train_batch)
				class_probs = self.softmax(fc_output)

				loss = self.get_CE_loss(class_probs, label_batch)
				self.backward(loss)
				epoch_loss += loss

				print("Batch Iter: %4d/%4d, \tEpoch: %2d/%2d, \tLoss: %.4f, \tlr: %.2e\n" % (batch+1, train_len, epoch+1, epochs, loss, self.lr))

			epoch_error[epoch] = epoch_loss
			if has_loss_converged(prev_epoch_loss, epoch_loss):
				self.lr_decay()

		print("Training complete. Saving weights...")
		np.save(activation_fn+'_MNIST', np.asarray([self.weights, self.bias]))

	def predict(self, test_data):
		"""
		Return class probabilites for test data.
		Expects preprocessed data!
		"""
		# raise NotImplementedError
		return self.softmax(self.forward(test_data))

	def score(self, test_data, test_labels):
		"""
		Return prediction accuracy.
		Expects preprocessed data
		"""		
		# raise NotImplementedError
		return np.sum(test_labels==np.argmax(np.predict(test_data), axis=0))/len(test_labels)

	def relu(self, x):
		"""
		Return ReLU activation
		"""
		# raise NotImplementedError
		# if x > 0:
		# 	return x
		# return 0
		return np.maximum(x, 0)

	def sigmoid(self, x):
		"""
		Return sigmoid activation
		"""
		# raise NotImplementedError
		return 1/(1 + np.exp(-1*x))

	def linear(self, x, c=0.5):
		"""
		Return Linear activation. Chose 
		c parameter arbitrarily
		"""
		# raise NotImplementedError
		return c*x

	def tanh(self, x):
		"""
		Return Tanh activation
		"""
		# raise NotImplementedError
		return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

	def softmax(self, x):
		"""
		Return Softmax activation
		"""
		# raise NotImplementedError
		total = (np.sum(np.exp(x)))
		return np.exp(x)/total

	def d_relu(self, x):
		"""
		Return derivative of ReLU. If 
		y = f(x) then dy/dx = f'(x)
		"""
		# raise NotImplementedError
		temp = np.copy(x)
		temp[x<=0] = 0
		temp[x>0] = 1
		return temp

	def d_sigmoid(self, x):
		"""
		Return derivative of Sigmoid
		"""
		# raise NotImplementedError
		return self.sigmoid(x)*(1 - self.sigmoid(x))

	def d_linear(self, x, c=0.5):
		"""
		Return derivative of Linear
		"""
		# raise NotImplementedError
		return c

	def d_tanh(self, x):
		"""
		Return derivative of Tanh
		"""
		# raise NotImplementedError
		return  1 - np.square(self.tanh(x))

	def d_softmax(self, x, label_index, var_index):
		"""
		Return derivative of Softmax
		"""
		# raise NotImplementedError
		k_delta = 0
		if label_index==var_index: k_delta = 1
		return self.softmax(x, label_index)*(k_delta - self.softmax(x, var_index))

	def lr_decay(self):
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

	def get_CE_loss(self, soft_fc_output, truth_dist, batch_size=None):
		"""
		Return cross entropy loss. Assuming batch to be 
		something like mxn shape.
		"""
		# raise NotImplementedError
		temp = np.multiply(truth_dist, np.log(soft_fc_output+1e-12))
		batch_loss = -1 * np.sum(temp)/self.batch_size
		sample_wise_loss = -1 * np.max(temp)/self.batch_size
		return batch_loss, sample_wise_loss

	def d_nll(self, x):
		"""
		Return derivative of Negative Log Likelihood Loss.
		"""
		return -1 * 1/np.exp(-1*x)

	def forward(self, batch):
		"""
		Return batch logits output of network
		"""
		# raise NotImplementedError
		self.forward_list = list()
		for i in range(self.layers):
			forward_list.append((batch, self.weights[i], self.bias[i]))
			batch = self.activation_fn(np.dot(batch, self.weights[i].T) + self.bias[i])

		return batch

	def backward(self, sample_wise_loss):
		"""
		Update weights and biases!
		General stucture:
		CE_loss <- Softmax_FC <- Logits_FC <- Weights
		"""
		raise NotImplementedError
		# Return d_nll for correct class for each sample!
		self.weight_update = dict()
		self.bias_update = dict()

		d_CELoss_SoftmaxFC = d_nll(sample_wise_loss)
		d_CELoss_LogitsFC = d_CELoss_SoftmaxFC * d_softmax(d_CELoss_SoftmaxFC)
		temp = np.copy(d_CELoss_LogitsFC)

		for i in range(self.layers-1, 0, -1):
			temp = temp * self.d_activation_fn(temp)
			self.weight_update[i] = self.lr * temp * self.forward_list[i][0]
			bias_update = temp
			self.bias_update[i] = self.lr* temp


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
