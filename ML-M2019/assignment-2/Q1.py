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

def extract_dataset_4_5():
	x_data = list()
	y_data = list()
	for file in os.listdir('./ml_ass_2_data'):
		h5 = h5py.File('./ml_ass_2_data/'+file, 'r')
		if file!='data_4.h5' and file!='data_5.h5':
			continue
		x_data = h5.get('x').value
		y_data = h5.get('y').value
		h5.close()

	# print(x_data, y_data)
	# print(np.asarray(x_data).shape, np.asarray(y_data).shape)
	return np.asarray(x_data), np.asarray(y_data)

def homebrew_kernel(X_, Z_):
	"""
	Homebrewed RBF kernel
	"""

	X = np.copy(X_)
	Z = np.copy(Z_)

	r = list(X.shape)[0]
	s = list(Z.shape)[0]
	tex = np.ones(shape=(r, s))
	for i, x in enumerate(X):
		for j, z in enumerate(Z):
			tex[i][j] = np.linalg.norm(x - z)

	np.exp(-1*np.square(tex))

	return tex

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

	"""
	Using code from here: 
	https://scikit-learn.org/0.18/auto_examples/svm/plot_iris.html
	"""

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
	"""
	For reference:
	Scikit kernel score: 0.8775
	Homebrewed kernel score: 0.24
	"""
	from sklearn import svm
	from sklearn.utils import shuffle
	from sklearn.metrics import accuracy_score

	X, y = extract_dataset_4_5()
	X, y = shuffle(X, y)
	split = 4*len(X)//5
	X_train, X_test = X[:split], X[split:]
	y_train, y_test = y[:split], y[split:]

	# clf_scikit = svm.SVC(C=1.0, gamma='scale', kernel='rbf')  
	clf_homebrew = svm.SVC(C=1.0, kernel=homebrew_kernel)

	# clf_scikit.fit(np.copy(X_train), np.copy(y_train))
	clf_homebrew.fit(np.copy(X_train), np.copy(y_train))

	# preds_scikit = clf_scikit.predict(np.copy(X_test))
	preds_homebrew = clf_homebrew.predict(np.copy(X_test))

	# print("Scikit kernel score:", accuracy_score(np.copy(y_test), preds_scikit))
	print("Homebrewed kernel score:", accuracy_score(np.copy(y_test), preds_homebrew))


####################	
#	Main	
####################

if __name__ == '__main__':
	# Uncomment question to run.
	# extract_h5_files()
	# q1_a()
	# q1_b()
	# q1_c()
	q1_d()