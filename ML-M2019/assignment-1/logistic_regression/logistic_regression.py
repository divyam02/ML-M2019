import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

################################################
#	Questions
################################################

def binary_logistic_regression(full_dataset, lr=1e-3, epochs=100,
								L1=False, L2=False, plot=False, reg_param=1):
	"""
	"""
	accuracy_train_L1 = np.zeros((epochs))
	accuracy_train_L2 = np.zeros((epochs))
	accuracy_test = np.zeros((epochs))

	weights_L1 = init_weights(shape=(105, 1))
	weights_L2 = init_weights(shape=(105, 1))

	real_train_truth = np.copy(full_dataset[-1, :])

	full_dataset, normalize_np = normalize(full_dataset)
	train_dataset = np.copy(full_dataset[:-1, :])
	train_truth = np.copy(full_dataset[-1])
	training_size = list(train_dataset.shape)[1]

	for epoch in range(epochs):
		for sample in range(training_size):
			x = train_dataset[:, sample].reshape((105, 1))

			h_L1 = np.dot(np.transpose(weights_L1), x)
			cost_L1 = h_L1 - train_truth[sample]
			weights_L1 = weight_update_reg(cost_L1, x, weights_L1, lr, True, False, reg_param)

			h_L2 = np.dot(np.transpose(weights_L2), x)
			cost_L2 = h_L2 - train_truth[sample]
			weights_L2 = weight_update_reg(cost_L2, x, weights_L2, lr, False, True, reg_param)

		accuracy_train_L1[epoch] = test_accuracy(train_dataset, real_train_truth, weights_L1)
		accuracy_train_L2[epoch] = test_accuracy(train_dataset, real_train_truth, weights_L2)

		print("L1 acc:" ,accuracy_train_L1[epoch])
		print("L2 acc:", accuracy_train_L2[epoch])

	plot(accuracy_train_L1, accuracy_train_L2)

		#accuracy_test[epoch] = test_accuracy()

################################################
#	Auxillary Methods
################################################

def test_accuracy(train_dataset, train_truth, weights):
	pred = 1/(1 + np.exp(-1*np.dot(np.transpose(train_dataset), weights)))
	pred[pred>=0.5] = 1
	pred[pred<0.5] = 0
	print(pred.shape)
	return np.sum(np.transpose(pred)[0]==np.transpose(train_truth)[0])/30162


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

def init_weights(shape=(11, 1)):
	"""
	Initialize weights for linear regression.
	choose weights from gaussian(0, 1)
	"""
	weights = np.random.normal(0, 0.5, size=shape)
	return weights

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

def gradient_descent():
	"""
	"""
	pass

def plot(L1_accuracy, L2_accuracy):
	"""
	Plots for training and validation
	sets, over 5 manifolds
	"""
	total_epochs = 100
	plt.figure(figsize=(20, 10))
	plt.plot(range(total_epochs), L1_accuracy, label="L1 Logistic Regression")
	plt.plot(range(total_epochs), L2_accuracy, label="L2 Logistic Regression")

	plt.savefig("Accuracy_plots.png")
	plt.show()


def feature_extraction(data_file):
	"""
	Extract meaningful features from 
	the provided data. Do separately for
	training and validation data!
	"""
	df = pd.read_csv(data_file)
	a = pd.get_dummies(df)
	full_dataset = a.to_numpy()
	full_dataset = np.transpose(full_dataset)
	print(full_dataset)
	a = full_dataset[-1, :]
	a = np.reshape(a, (1, 30162))
	full_dataset = np.concatenate((full_dataset[:-2, :], a), axis=0)
	full_dataset = np.concatenate((np.ones((1, 30162)), full_dataset), axis=0)
	# If greater than 50k value is one.
	print(full_dataset)
	return full_dataset


################################################
#	Main
################################################

if __name__ == '__main__':
	full_dataset = feature_extraction(data_file="./train.csv")
	binary_logistic_regression(full_dataset)
