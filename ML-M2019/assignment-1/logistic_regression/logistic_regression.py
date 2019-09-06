import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

################################################
#	Questions
################################################


# IMPORTANT!
# Idiot! h = g(x)! No wonder this was bouncing around
# 50% accuracy. It was making a hyperplane for prediction!
# Fix this!


def binary_logistic_regression(full_dataset_, lr=1e-5, epochs=50,
								L1=False, L2=False, plot=False, reg_param=0.01):
	"""
	Training accuracy for L1: ~ 0.80
	Training accuracy for L2: ~ 0.75

	Validation accuracy for L1 ~ 0.42  
	Validation accuracy for L2 ~ 0.45
	
	Testing accuracy for L1 ~ 0.30
	Testing accuracy for L2 ~ 0.35

	L2 regularization is performing better during testing due to smaller weight updates
	as iterations occur, which is also the reason why L1 regularization performs better
	with a constant step update.
	"""
	accuracy_train_L1 = np.zeros((5, 50))
	accuracy_train_L2 = np.zeros((5, 50))
	accuracy_validation_L1 = np.zeros((5, 50))
	accuracy_validation_L2 = np.zeros((5, 50))
	accuracy_test_L1 = np.zeros((5, 50))
	accuracy_test_L2 = np.zeros((5, 50))

	for i in range(5):
		full_dataset = np.copy(full_dataset_)
		full_dataset = np.transpose(full_dataset)
		np.random.shuffle(full_dataset)
		full_dataset = np.transpose(full_dataset)


		mark = 30162//5
		train_dataset_ = np.copy(full_dataset[:, :mark*3])
		val_dataset_ = np.copy(full_dataset[:, mark*3:mark*4])
		test_dataset_ = np.copy(full_dataset[:, mark*4:])

		real_train_truth = np.copy(train_dataset_[-1])
		real_validation_truth = np.copy(val_dataset_[-1])
		real_test_truth = np.copy(test_dataset_[-1])

		train__dataset, normalize_np = normalize(train_dataset_)
		val__dataset = val_normalize(normalize_np, val_dataset_)
		test__dataset = val_normalize(normalize_np, test_dataset_)

		train_dataset = np.copy(train__dataset[:-1, :])
		val_dataset = np.copy(val__dataset[:-1, :])
		test_dataset = np.copy(test__dataset[:-1, :])

		training_size = list(train_dataset.shape)[1]
		weights = init_weights(shape=(105, 1))

		print(train_dataset.shape)
		print(val_dataset.shape)
		print(test_dataset.shape)

		L1 = True
		L2 = False

		for epoch in range(epochs):
			for sample in range(training_size):

				x = train_dataset[:, sample].reshape((105, 1))
				h = 1.0/(1 + np.exp(-1 * np.dot(np.transpose(weights), x)))
				cost = h - real_train_truth[sample]
				weights = weight_update_reg(np.copy(cost), np.copy(x), np.copy(weights), lr, L1, L2, reg_param)

			train_test = test_accuracy(np.copy(train_dataset), np.copy(real_train_truth), np.copy(weights))
			val_test = test_accuracy(np.copy(val_dataset), np.copy(real_validation_truth), np.copy(weights))
			test_test = test_accuracy(np.copy(test_dataset), np.copy(real_test_truth), np.copy(weights))
			accuracy_train_L1[i, epoch] = train_test
			accuracy_validation_L1[i, epoch] = val_test
			accuracy_test_L1[i, epoch] = test_test

			print("L1 train acc:" ,accuracy_train_L1[i, epoch])
			print("L1 val acc:", accuracy_validation_L1[i, epoch])
			print("L1 test acc:", accuracy_test_L1[i, epoch], "\n")

		weights = init_weights(shape=(105, 1))
		L1 = False
		L2 = True
	 
		for epoch in range(epochs):
			for sample in range(training_size):

				x = train_dataset[:, sample].reshape((105, 1))
				h = 1.0/(1 + np.exp(-1*np.dot(np.transpose(weights), x)))
				cost = h - real_train_truth[sample]
				weights = weight_update_reg(np.copy(cost), np.copy(x), np.copy(weights), lr, L1, L2, reg_param)
				# if sample % 1000 == 0:
			train_test = test_accuracy(np.copy(train_dataset), np.copy(real_train_truth), np.copy(weights))
			val_test = test_accuracy(np.copy(val_dataset), np.copy(real_validation_truth), np.copy(weights))
			test_test = test_accuracy(np.copy(test_dataset), np.copy(real_test_truth), np.copy(weights))
			accuracy_train_L2[i, epoch] = train_test
			accuracy_validation_L2[i, epoch] = val_test
			accuracy_test_L2[i, epoch] = test_test

			print("L2 train acc:" ,accuracy_train_L2[i, epoch])
			print("L2 val acc:", accuracy_validation_L2[i, epoch])
			print("L2 test acc:", accuracy_test_L2[i, epoch], "\n")

	accuracy_plot(np.mean(accuracy_train_L1, axis=0), np.mean(accuracy_train_L2, axis=0))
	input("Continue?")
	accuracy_plot(np.mean(accuracy_validation_L1, axis=0), np.mean(accuracy_validation_L2, axis=0))
	input("Continue?")
	accuracy_plot(np.mean(accuracy_test_L1, axis=0), np.mean(accuracy_test_L2, axis=0))

def multi_logistic_regression():
	"""
	Using SAGA because of sparse matrices!

	L1 training accuracy: 0.889
	L1 testing accuracy: 0.892

	L2 training accuracy: 0.889
	L2 testing accuracy: 0.891
	"""
	from mnist import MNIST
	data = MNIST("./MNIST")
	train_images, train_labels = data.load_training()
	test_images, test_labels = data.load_testing()

	train_images = np.reshape(np.asarray(train_images), (60000, 784))
	train_labels = np.reshape(np.asarray(train_labels), (60000,))
	test_images = np.reshape(np.asarray(test_images), (10000, 784))
	test_labels = np.reshape(np.asarray(test_labels), (10000,))

	from sklearn.preprocessing import StandardScaler
	from sklearn.linear_model import LogisticRegression

	scaler = StandardScaler()
	X_train = scaler.fit_transform(train_images)
	X_test = scaler.fit_transform(test_images)

	# For L1 regularization
	clf = LogisticRegression(penalty='l1', solver='saga',
							 multi_class='multinomial', tol=0.1)
	clf.fit(X_train, train_labels)
	print("L1 regularized train accuracy:", clf.score(X_train, train_labels))
	print("L1 regularized test accuracy:", clf.score(X_test, test_labels), "\n")

	# For L2 regularization
	clf = LogisticRegression(penalty='l2', solver='saga',
							 multi_class='multinomial', tol=0.1)
	clf.fit(X_train, train_labels)
	print("L2 regularized train accuracy:", clf.score(X_train, train_labels))
	print("L2 regularized test accuracy:", clf.score(X_test, test_labels))

################################################
#	Auxillary Methods
################################################

def val_normalize(normalize_np, val_dataset):
	for i in range(len(val_dataset)):
		val_dataset[i] = (val_dataset[i] - normalize_np[i, 0])/(normalize_np[i, 1] + 1e-6)

	return val_dataset

def test_accuracy(train_dataset, train_truth, weights):
	pred = 1 + np.exp(-1*np.dot(np.transpose(train_dataset), weights))
	pred[pred<2] = 1
	pred[pred>=2] = 0

	return np.sum(np.transpose(pred)[0]!=np.transpose(train_truth)[0])/len(pred)



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
	weights = np.zeros(shape)
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
		train_data[i] = (train_data[i]-row_mean)/(row_std + 1e-6)

		normalize_np[i, 0], normalize_np[i, 1] = np.copy(row_mean), np.copy(row_std)

	normalize_np[0, 1] = 1
	return train_data, normalize_np

def gradient_descent():
	"""
	"""
	pass

def accuracy_plot(L1_accuracy, L2_accuracy):
	"""
	Plots for training and validation
	sets, over 5 manifolds
	"""
	plt.figure(figsize=(20, 10))
	plt.plot(range(len(L1_accuracy)), L1_accuracy, label="L1 Logistic Regression")
	plt.plot(range(len(L2_accuracy)), L2_accuracy, label="L2 Logistic Regression")
	plt.legend(loc='lower right')
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
	# full_dataset = feature_extraction(data_file="./train.csv")
	# binary_logistic_regression(full_dataset)
	multi_logistic_regression()