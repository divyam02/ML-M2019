import sklearn
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

####################
#	Methods
####################

def extract_h5_files():
	"""
	Data dimensions: 2000x2
	"""
	x_data = list()
	y_data = list()
	for file in os.listdir('./ml_ass_2_data'):
		h5 = h5py.File('./ml_ass_2_data/'+file, 'r')
		x_data = h5.get('x').value
		y_data = h5.get('y').value
		h5.close()

	# print(x_data, y_data)
	# print(np.asarray(x_data).shape, np.asarray(y_data).shape)
	return np.asarray(x_data), np.asarray(y_data)

####################	
#	Questions
####################

def q1_a():
	X, y = extract_h5_files()

	print(y)
	A = np.argwhere(y==0)
	B = np.argwhere(y==1)	

	x_0 = list()
	y_0 = list()
	x_1 = list()
	y_1 = list()

	for z in A:
		x, y = X[z][0]
		x_0.append(x)
		y_0.append(y)

	for z in B:
		x, y = X[z][0]
		x_1.append(x)
		y_1.append(y)

	plt.figure(figsize=(20, 10))
	plt.scatter(x_0, y_0, color="blue", edgecolors="white")
	plt.scatter(x_1, y_1, color="red", edgecolors="white")
	plt.xticks(())
	plt.yticks(())
	plt.title('Scatter plot for given data')
	plt.savefig('Q1_a')

def q1_b():
	from sklearn import svm

	X, y = extract_h5_files()
	clf = svm.SVC(gamma='scale')
	clf.fit(X, y)
	h = 0.02
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
							np.arange(y_min, y_max, h))

	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.figure(figsize=(20, 10))

	plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

	# Plot also the training points
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='black')
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.xticks(())
	plt.yticks(())
	plt.title('SVM decision boundary using RBF kernel')
	plt.savefig('Q1_b')

def q1_c():
	from sklearn import svm

	X, y = extract_h5_files()
	clf = svm.SVC(gamma='scale')
	clf.fit(X, y)
	preds = clf.predict(X)
	# print(y)
	# print(y.shape)	
	mis = preds==y
	print(mis)
	print(mis.shape)
	A = np.argwhere(y==0)
	B = np.argwhere(y==1)	

	x_0 = list()
	y_0 = list()
	x_1 = list()
	y_1 = list()

	for z in A:
		if mis[z]==True:
			x, y = X[z][0]
			x_0.append(x)
			y_0.append(y)

	for z in B:
		if mis[z]==True:
			x, y = X[z][0]
			x_1.append(x)
			y_1.append(y)

	plt.figure(figsize=(20, 10))
	plt.scatter(x_0, y_0, color="blue", edgecolors="white")
	plt.scatter(x_1, y_1, color="red", edgecolors="white")
	plt.xticks(())
	plt.yticks(())
	plt.title('Scatter plot with noise removed')
	plt.savefig('Q1_c')


def q1_d():
	pass


####################	
#	Main	
####################

if __name__ == '__main__':
	# Uncomment question to run.
	# extract_h5_files()
	# q1_a()
	# q1_b()
	q1_c()
	# q1_d()