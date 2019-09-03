import numpy as np
import matplotlib.pyplot as plt
import argparse

#################################################
#	Questions
#################################################

def plot_RMSE(mean_train_RMSE, mean_val_RMSE):
	"""
	Plots for training and validation
	sets, over 5 manifolds
	"""
	total_epochs = 100
	fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 20))
	ax[0].plot(range(total_epochs), mean_train_RMSE, label="Training Loss")
	ax[1].plot(range(total_epochs), mean_val_RMSE, label="Validation Loss")

	ax[0].set(title = "Training Loss")
	ax[0].set(xlabel = "Epochs")
	ax[0].set(ylabel = "RMSE")

	ax[1].set(title = "Validation Loss")
	ax[1].set(xlabel = "Epochs")
	ax[1].set(ylabel = "RMSE")

	#plt.savefig("RMSE_plots.png")
	plt.show()

def linear_regression_solve(full_dataset, lr=0.001, stop=1e-5, folds=5,
							epochs=100):
	val_size = list(full_dataset.shape)[1]//folds

	RMSE_train_values = np.zeros((folds, epochs))
	RMSE_val_values = np.zeros((folds, epochs))


	for i in range(folds):
		"""
		Get the requisite training and 
		validation sets.
		"""
		# Initialize weights every fold!
		weights = init_weights()

		val_dataset = full_dataset[:, i*val_size:(i+1)*val_size]

		train_dataset = np.concatenate((full_dataset[:, 0:i*val_size], 
										full_dataset[:, (i+1)*val_size:]), axis=1)

		train_dataset, normalize_np = normalize(train_dataset)
		val_dataset = val_normalize(normalize_np, val_dataset)

		val_truth = val_dataset[-1]
		val_dataset = val_dataset[:-1, :]

		train_truth = train_dataset[-1]
		train_dataset = train_dataset[:-1, :]

		prev_loss = np.inf
		training_size = list(train_dataset.shape)[1]

		for epoch in range(epochs):
			"""
			Train on data!
			Update weights!
			"""
			for sample in range(training_size):

				x = train_dataset[:, sample].reshape((11, 1))
				h = np.dot(np.transpose(weights), x)
				cost = h - train_truth[sample]
				weights = weight_update(cost, x, weights, lr)
				RMSE_train_values[i, epoch] += np.square(cost)

			RMSE_train_values[i, epoch] = np.sqrt(RMSE_train_values[i, epoch]/training_size)

			"""
			Test on validation set!
			Do not update weights!
			Use calculated mean and std!
			"""
			for sample in range(val_size):

				x = val_dataset[:, sample].reshape((11, 1))
				h = np.dot(np.transpose(weights), x)
				cost = h - val_truth[sample]
				RMSE_val_values[i, epoch] += np.square(cost)

			RMSE_val_values[i, epoch] = np.sqrt(RMSE_val_values[i, epoch]/val_size)

			# print("Finished fold:", i, " epoch:", epoch)
			# print("RMSE Train:", RMSE_train_values[i, epoch])
			# print("RMSE Validation:", RMSE_val_values[i, epoch], "\n")

	mean_train_RMSE = np.mean(RMSE_train_values, axis=0)
	mean_val_RMSE = np.mean(RMSE_val_values, axis=0)

	print("Mean RMSE value for training sets:", mean_train_RMSE[-1])
	print("Mean RMSE value for validation sets:", mean_val_RMSE[-1])

	#plot_RMSE(mean_train_RMSE, mean_val_RMSE)


def normal_eqn_solve(full_dataset, folds=5):
	"""
	Directly get optimal parameters, 
	get RMSE for 5 folds
	"""

	print("Solving using Normal Equations...\n")

	val_size = list(full_dataset.shape)[1]//folds

	RMSE_train_values = np.zeros((folds, 1))
	RMSE_val_values = np.zeros((folds, 1))

	for i in range(folds):

		val_dataset = full_dataset[:, i*val_size:(i+1)*val_size]

		train_dataset = np.concatenate((full_dataset[:, 0:i*val_size], 
										full_dataset[:, (i+1)*val_size:]), axis=1)

		train_dataset, normalize_np = normalize(train_dataset)
		val_dataset = val_normalize(normalize_np, val_dataset)

		val_truth = val_dataset[-1]
		val_dataset = val_dataset[:-1, :]

		train_truth = train_dataset[-1]
		train_dataset = train_dataset[:-1, :]

		xTx_inv = np.linalg.inv(np.dot(train_dataset, np.transpose(train_dataset)))
		optimal_weights = np.dot(np.dot(xTx_inv, train_dataset), train_truth)

		RMSE_train_values[i] = np.sqrt(np.mean(np.square(np.dot(np.transpose(train_dataset), optimal_weights))))
		# print("RMSE values for training set:")
		# print("fold:", i+1, "value:", RMSE_train_values[i])


		RMSE_val_values[i] = np.sqrt(np.mean(np.square(np.dot(np.transpose(val_dataset), optimal_weights))))
		# print("RMSE values for validation set:")
		# print("fold:", i+1, "value:", RMSE_val_values[i], "\n")

	print("Mean RMSE for training sets:", np.mean(RMSE_train_values))
	print("Mean RMSE for validation sets:", np.mean(RMSE_val_values))


def comparisons():
	"""
	Compare RMSE values, add some plots
	"""
	pass

def regularization():
	pass

#################################################
#	Auxillary Methods
#################################################

def weight_update(actual_cost_for_sample, curr_sample, curr_weights,
				  lr):
	"""
	Gradient descent parameter updates.
	Use stochastic gradient descent.

	@cost:
		(h(x) - y)
	@curr_sample:
		sample for which cost was calculated
		must be a vector!
	@curr_weights:
		current weights being used
	@lr:
		current learning rate

	Returns updated weights
	"""
	new_weights = curr_weights - (lr * (actual_cost_for_sample) * curr_sample)

	return new_weights


def val_normalize(normalize_np, val_dataset):
	for i in range(len(val_dataset)):
		val_dataset[i] = (val_dataset[i] - normalize_np[i, 0])/normalize_np[i, 1]

	return val_dataset

def normalize(train_data):
	"""
	Normalize features by row mean and std_dev.
	Keep for future usage!

	Returns normalized data and normalization
	parameters.
	"""
	# Keep track for feature and mean, std
	normalize_np = np.zeros((len(train_data), 2))
	for i in range(1, len(train_data)):

		row_mean  = np.mean(train_data[i])
		row_std = np.std(train_data[i])
		train_data[i] = (train_data[i]-row_mean)/row_std

		normalize_np[i, 0], normalize_np[i, 1] = row_mean, row_std

	normalize_np[0, 1] = 1
	return train_data, normalize_np


def init_weights(shape=(11, 1)):
	"""
	Initialize weights for linear regression.
	choose weights from gaussian(0, 1)
	"""
	weights = np.random.normal(0, 0.5, size=shape)

	return weights


def extract_features(data_file):
	"""
	Extract meaningful features from 
	the provided data. Do separately for
	training and validation data!
	"""
	full_dataset = None

	with open(data_file, 'r') as f:
		for file in f.readlines():

			a = file.split()
			temp_np = np.asarray(a[1:], dtype=np.float32)
			"""
			Use one-hot encoding for sex parameter. 
			Also add extra term to account for model
			bias.
			"""
			if a[0]=='I':
				temp_np = np.concatenate((np.array((1, 1, 0, 0), dtype=np.float32), temp_np), axis=0)
			elif a[0]=='M':
				temp_np = np.concatenate((np.array((1, 0, 1, 0), dtype=np.float32), temp_np), axis=0)
			else:
				temp_np = np.concatenate((np.array((1, 0, 0, 1), dtype=np.float32), temp_np), axis=0)

			temp_np = np.reshape(temp_np, (12, 1))

			try:
				full_dataset = np.concatenate((full_dataset, temp_np), axis=1)
			except:
				full_dataset = temp_np

		# print(full_dataset)
		# print(full_dataset.shape)
		# print(np.transpose(full_dataset))
		# print(np.transpose(full_dataset).shape)
		# print(np.transpose(full_dataset)[0])
		# print(full_dataset[:, 0])
	return full_dataset

#################################################
#	Main Method
#################################################

if __name__ == '__main__':
	full_dataset = extract_features(data_file='./abalone/Dataset.data')
	linear_regression_solve(full_dataset)
	normal_eqn_solve(full_dataset)