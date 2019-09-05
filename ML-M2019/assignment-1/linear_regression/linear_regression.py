import numpy as np
import matplotlib.pyplot as plt
import argparse

#################################################
#	Questions
#################################################

def test_linear_regression(test_dataset, train_dataset, weights, reg_param=0.0001,lr=1e-3, L1=False,
												L2=True, folds=5, epochs=1,
												grid_search=False, plot=False):
	__, normalize_np = normalize(train_dataset)

	RMSE_train_values = 0
	# train_dataset = np.copy(test_dataset)
	# train_dataset, normalize_np = normalize(train_dataset)
	train_dataset = val_normalize(np.copy(test_dataset), normalize_np)
	train_truth = train_dataset[-1]
	train_dataset = train_dataset[:-1, :]

	training_size = list(train_dataset.shape)[1]

	for epoch in range(epochs):

		for sample in range(training_size):

			x = train_dataset[:, sample].reshape((11, 1))
			h = np.dot(np.transpose(weights), x)
			cost = h - train_truth[sample]
			weights = weight_update_reg(cost, x, weights, lr, L1, L2, reg_param)
			if L1:
				reg_cost = reg_param * np.sum(np.abs(weights))
			elif L2:
				reg_cost = reg_param * np.sum(np.square(weights))
			else:
				reg_cost = 0

			RMSE_train_values += np.square(cost) + reg_cost

		RMSE_train_values = np.sqrt(RMSE_train_values/training_size)

	print("L1, L2:", L1, L2, "Mean test RMSE:", RMSE_train_values)

def linear_regression_with_regularization_solve(full_dataset, reg_param, lr=1e-3, L1=False,
												L2=True, folds=5, epochs=100,
												grid_search=False, plot=False, get_optimal_weights=False):

	"""
	L1 RMSE training/validation/test: 0.678, 0.712, 0.583
	L2 RMSE training/validaton/test: 0.678, 0.711, 0.464
	"""
	val_size = list(full_dataset.shape)[1]//folds

	RMSE_train_values = np.zeros((folds, epochs))
	RMSE_val_values = np.zeros((folds, epochs))

	for i in range(folds):

		# Initialize weights every fold!
		weights = init_weights()
		val_dataset = np.copy(full_dataset[:, i*val_size:(i+1)*val_size])
		train_dataset = np.copy(np.concatenate((full_dataset[:, 0:i*val_size], 
										full_dataset[:, (i+1)*val_size:]), axis=1))

		train_dataset, normalize_np = normalize(train_dataset)
		val_dataset = val_normalize(normalize_np, val_dataset)

		val_truth = val_dataset[-1]
		val_dataset = val_dataset[:-1, :]

		train_truth = train_dataset[-1]
		train_dataset = train_dataset[:-1, :]

		prev_loss = np.inf
		training_size = list(train_dataset.shape)[1]

		for epoch in range(epochs):

			for sample in range(training_size):

				x = train_dataset[:, sample].reshape((11, 1))
				h = np.dot(np.transpose(weights), x)
				cost = h - train_truth[sample]
				weights = weight_update_reg(cost, x, weights, lr, L1, L2, reg_param)
				if L1:
					reg_cost = reg_param * np.sum(np.abs(weights))
				elif L2:
					reg_cost = reg_param * np.sum(np.square(weights))
				else:
					reg_cost = 0

				RMSE_train_values[i, epoch] += np.square(cost) + reg_cost

			RMSE_train_values[i, epoch] = np.sqrt(RMSE_train_values[i, epoch]/training_size)

			for sample in range(val_size):

				x = val_dataset[:, sample].reshape((11, 1))
				h = np.dot(np.transpose(weights), x)
				cost = h - val_truth[sample]
				if L1:
					reg_cost = reg_param * np.sum(np.abs(weights))
				elif L2:
					reg_cost = reg_param * np.sum(np.square(weights))
				else:
					reg_cost = 0
				RMSE_val_values[i, epoch] += np.square(cost) + reg_cost

			RMSE_val_values[i, epoch] = np.sqrt(RMSE_val_values[i, epoch]/val_size)

			# print("Finished fold:", i, " epoch:", epoch)
			# print("RMSE Train:", RMSE_train_values[i, epoch])
			# print("RMSE Validation:", RMSE_val_values[i, epoch], "\n")

	mean_train_RMSE = np.mean(RMSE_train_values, axis=0)
	mean_val_RMSE = np.mean(RMSE_val_values, axis=0)

	print("L1:", L1, "L2:", L2, "reg_param:", reg_param)
	print("Mean RMSE value for training sets:", mean_train_RMSE[-1])
	print("Mean RMSE value for validation sets:", mean_val_RMSE[-1])

	if grid_search:
		return reg_param, mean_val_RMSE[-1]

	if plot:
		plot_RMSE(mean_train_RMSE, mean_val_RMSE)

	if get_optimal_weights:
		return np.copy(weights)



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

		val_dataset = np.copy(full_dataset[:, i*val_size:(i+1)*val_size])

		train_dataset = np.copy(np.concatenate((full_dataset[:, 0:i*val_size], 
										full_dataset[:, (i+1)*val_size:]), axis=1))

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
				weights = weight_update_reg(cost, x, weights, lr, L1, L2, reg_param)
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

			print("Finished fold:", i, " epoch:", epoch)
			print("RMSE Train:", RMSE_train_values[i, epoch])
			print("RMSE Validation:", RMSE_val_values[i, epoch], "\n")

	mean_train_RMSE = np.mean(RMSE_train_values, axis=0)
	mean_val_RMSE = np.mean(RMSE_val_values, axis=0)

	print("Mean RMSE value for training sets:", mean_train_RMSE[-1])
	print("Mean RMSE value for validation sets:", mean_val_RMSE[-1])

	plot_RMSE(mean_train_RMSE, mean_val_RMSE)


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

		val_dataset = np.copy(full_dataset[:, i*val_size:(i+1)*val_size])

		train_dataset = np.copy(np.concatenate((full_dataset[:, 0:i*val_size], 
										full_dataset[:, (i+1)*val_size:]), axis=1))

		train_dataset, normalize_np = normalize(train_dataset)
		val_dataset = val_normalize(normalize_np, val_dataset)

		val_truth = val_dataset[-1]
		val_dataset = val_dataset[:-1, :]

		train_truth = train_dataset[-1]
		train_dataset = train_dataset[:-1, :]

		xTx_inv = np.linalg.inv(np.dot(train_dataset, np.transpose(train_dataset)))
		optimal_weights = np.dot(np.dot(xTx_inv, train_dataset), train_truth)

		RMSE_train_values[i] = np.sqrt(np.mean(np.square(np.dot(np.transpose(train_dataset), optimal_weights))))
		print("RMSE values for training set:")
		print("fold:", i+1, "value:", RMSE_train_values[i])


		RMSE_val_values[i] = np.sqrt(np.mean(np.square(np.dot(np.transpose(val_dataset), optimal_weights))))
		print("RMSE values for validation set:")
		print("fold:", i+1, "value:", RMSE_val_values[i], "\n")

	print("Mean RMSE for training sets:", np.mean(RMSE_train_values))
	print("Mean RMSE for validation sets:", np.mean(RMSE_val_values))

	# Return lowest value index, for regularization!
	min_index = np.argmin(RMSE_val_values)

	val_dataset = np.copy(full_dataset[:, min_index*val_size:(min_index+1)*val_size])

	train_dataset = np.copy(np.concatenate((full_dataset[:, 0:min_index*val_size], 
									full_dataset[:, (min_index+1)*val_size:]), axis=1))

	train_dataset, normalize_np = normalize(train_dataset)
	val_dataset = val_normalize(normalize_np, val_dataset)

	# val_truth = val_dataset[-1]
	# val_dataset = val_dataset[:-1, :]

	# train_truth = train_dataset[-1]
	# train_dataset = train_dataset[:-1, :]

	return train_dataset, val_dataset

def comparisons():
	"""
	It is observed the mean RMSE values
	obtained by gradient descent are slightly 
	better (by a magnitude of order 1e-2) as 
	compared to RMSE values obtained by 
	Normal Equations
	"""
	pass

def regularization(full_dataset):
	"""
	Use regularization to do stuff!
	"""
	train_dataset, test_dataset = normal_eqn_solve(full_dataset)
	L1_optimal, L2_optimal = 1e-4, 1e-4
	# L1_optimal, L2_optimal = grid_search(np.copy(train_dataset))

	L1_optimal_weights = linear_regression_with_regularization_solve(np.copy(train_dataset), 
												L1=True, L2=False, reg_param=L1_optimal,
												plot=False, get_optimal_weights=True)
	input("Continue?")

	L2_optimal_weights = linear_regression_with_regularization_solve(np.copy(train_dataset),
												L2=True, L1=False, reg_param=L2_optimal,
												plot=False, get_optimal_weights=True)

	test_linear_regression(np.copy(test_dataset), train_dataset, L1_optimal_weights,L2=False, L1=True, reg_param=L1_optimal,
							plot=True)	

	input("Continue?")

	test_linear_regression(np.copy(test_dataset), train_dataset, L2_optimal_weights, L2=True, L1=False, reg_param=L2_optimal,
							plot=True)


def best_line_fit(data_file, epochs=100, lr=1e-3, reg_param=0.5):
	"""
	The lines are practically the same for the calculated optimal
	values L1 and L2 regularization, basically ~ 0. 
	
	For other values, regularization is actually reducing performance...
	"""
	full_dataset = None
	with open(data_file, 'r') as f:
		for line in f.readlines():

			try:
				a, b = map(float, line.split(","))
			except:
				print("Hmm..")
				continue

			temp_np = np.asarray([a, b])
			temp_np = np.reshape(temp_np, (2, 1))
			# Bias term:
			temp_np = np.concatenate((np.reshape(np.asarray((1)), (1,1)), temp_np), axis=0)
			try:
				full_dataset = np.concatenate((full_dataset, temp_np), axis=1)
			except:
				full_dataset = temp_np

	train_dataset = np.copy(full_dataset)
	colt_dataset = np.copy(full_dataset)

	# Line found by Vanilla SGD:	
	train_dataset, normalize_np = normalize(train_dataset)
	train_truth = train_dataset[-1]
	train_dataset = train_dataset[:-1, :]

	colt_truth = colt_dataset[-1, :]
	colt_dataset = colt_dataset[1, :]

	weights = init_weights(shape=(2, 1))

	training_size = list(train_dataset.shape)[1]

	for epoch in range(epochs):
		"""
		Train on data!
		Update weights!
		"""
		for sample in range(training_size):

			x = train_dataset[:, sample].reshape((2, 1))
			h = np.dot(np.transpose(weights), x)
			cost = h - train_truth[sample]
			weights = weight_update(cost, x, weights, lr)

	y_pts = np.asarray([(np.dot(np.transpose(weights), train_dataset[:, 0]) * normalize_np[-1, 1]) + normalize_np[-1, 0], 
						((h * normalize_np[-1, 1]) + normalize_np[-1, 0])])
	x_pts = np.asarray([(train_dataset[-1, 0] * normalize_np[1, 1]) + normalize_np[1, 0], 
						(x[-1, 0] * normalize_np[1, 1]) + normalize_np[1, 0]])
	y_pts = np.reshape(y_pts, (2,))
	
	plt.figure(figsize=(20, 10))
	plt.scatter(colt_dataset, colt_truth, color="black", edgecolors="white")
	plt.plot(x_pts, y_pts, label="Vanilla SGD")

	# Line found by L1 Regularized SGD
	weights =init_weights(shape=(2, 1))
	L1 = True
	L2 = False
	for epoch in range(epochs):

		for sample in range(training_size):

			x = train_dataset[:, sample].reshape((2, 1))
			h = np.dot(np.transpose(weights), x)
			cost = h - train_truth[sample]
			weights = weight_update_reg(cost, x, weights, lr, L1, L2, reg_param)

	y_pts = np.asarray([(np.dot(np.transpose(weights), train_dataset[:, 0]) * normalize_np[-1, 1]) + normalize_np[-1, 0], 
						((h * normalize_np[-1, 1]) + normalize_np[-1, 0])])
	x_pts = np.asarray([(train_dataset[-1, 0] * normalize_np[1, 1]) + normalize_np[1, 0], 
						(x[-1, 0] * normalize_np[1, 1]) + normalize_np[1, 0]])
	y_pts = np.reshape(y_pts, (2,))		
	plt.plot(x_pts, y_pts, label="L1 Regularized SGD")

	# Line found by L2 Regularized SGD
	weights =init_weights(shape=(2, 1))
	L1 = False
	L2 = True
	for epoch in range(epochs):

		for sample in range(training_size):

			x = train_dataset[:, sample].reshape((2, 1))
			h = np.dot(np.transpose(weights), x)
			cost = h - train_truth[sample]
			weights = weight_update_reg(cost, x, weights, lr, L1, L2, reg_param)

	y_pts = np.asarray([(np.dot(np.transpose(weights), train_dataset[:, 0]) * normalize_np[-1, 1]) + normalize_np[-1, 0], 
						((h * normalize_np[-1, 1]) + normalize_np[-1, 0])])
	x_pts = np.asarray([(train_dataset[-1, 0] * normalize_np[1, 1]) + normalize_np[1, 0], 
						(x[-1, 0] * normalize_np[1, 1]) + normalize_np[1, 0]])
	y_pts = np.reshape(y_pts, (2,))		
	plt.plot(x_pts, y_pts, label="L2 Regularized SGD")

	plt.ylabel("body weight")
	plt.xlabel("brain weight")
	plt.legend(loc="lower right")
	# plt.savefig("Best Line Fit.png")
	plt.show()

#################################################
#	Auxillary Methods
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

	plt.savefig("RMSE_plots.png")
	plt.show()


def grid_search(full_dataset):
	"""
	Grid search over target parameter...
	Return best value for selected metric! (RMSE)
	"""
	# For L1 regularization:

	L1_params = [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]

	min_param = None
	min_RMSE = np.inf

	for L1_param in L1_params:
		reg_param, reg_RMSE = linear_regression_with_regularization_solve(full_dataset, 
										L1=True, L2=False, reg_param=L1_param, grid_search=True)
		if reg_RMSE < min_RMSE:
			min_param = reg_param
			min_RMSE = reg_RMSE

	print("L1 Param:", min_param)
	L1_param = min_param


	L2_params = [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]

	min_param = None
	min_RMSE = np.inf

	for L2_param in L2_params:
		reg_param, reg_RMSE = linear_regression_with_regularization_solve(full_dataset, 
										L1=False, L2=True, reg_param=L2_param, grid_search=True)
		if reg_RMSE < min_RMSE:
			min_param = reg_param
			min_RMSE = reg_RMSE

	print("L2 Param:", min_param)
	L2_param = min_param

	return L1_param, L2_param

def weight_update_reg(actual_cost_for_sample, curr_sample, curr_weights,
					  lr, L1, L2, reg_param):
	"""
	Still using SGD!
	"""
	if L2:
		# print("doing L2!")
		new_weights = curr_weights - (lr * (actual_cost_for_sample * curr_sample) + 
								  	  lr * (reg_param * curr_weights))
	elif L1:
		# print("doing L1!")
		new_weights = curr_weights - (lr * (actual_cost_for_sample * curr_sample) +
									  lr * (reg_param * np.sign(curr_weights)))
	else:
		new_weights = curr_weights - (lr * (actual_cost_for_sample) * curr_sample)

	return new_weights


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
	new_weights = curr_weights - (lr * (actual_cost_for_sample * curr_sample))

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

		normalize_np[i, 0], normalize_np[i, 1] = np.copy(row_mean), np.copy(row_std)

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
	# linear_regression_solve(np.copy(full_dataset))
	# normal_eqn_solve(np.copy(full_dataset))
	# best_line_fit(data_file='./data.csv')
	# regularization(np.copy(full_dataset))