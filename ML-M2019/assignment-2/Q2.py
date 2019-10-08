import sklearn
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

####################
#	Methods
####################
def download_dataset():
	"""
	Download CIFAR-10 Dataset

	Done using wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
	"""
	pass

def get_dataset():
	import pickle
	x_data = list()
	y_data = list()
	files = ['data_batch_1', 
    		 'data_batch_2',
    		 'data_batch_3',
    		 'data_batch_4',
    		 'data_batch_5']

	for file in files:
		with open('./cifar_data/cifar-10-batches-py/'+file, 'rb') as fo:
			temp = pickle.load(fo, encoding='bytes')
			# print(temp.keys())
			x_data.append(temp[b'data'])
			y_data.append(temp[b'labels'])

	return np.asarray(x_data), np.asarray(y_data)

####################
#	Questions
####################

def no_kernel_svm():
	from sklearn import svm
	from sklearn import preprocessing
	from sklearn.utils import shuffle
	from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
	import pickle

	X_, y_ = get_dataset()
	"""
	
	"""
	# print(X)
	# print(y)
	# print(X.shape)
	# print(y.shape)
	X, y = np.copy(X_), np.copy(y_)

	# for i in range(X.shape[0]):
	# print("Fold:", str(i+1))
	temp_X = None
	temp_y = None

	for j in range(X.shape[0]):
		# if j==i:
		# 	continue
		try:
			temp_X = np.concatenate((temp_X, X[j]), axis=0)
			temp_y = np.concatenate((temp_y, y[j]), axis=0)
		except:
			temp_X = np.copy(X[j])
			temp_y = np.copy(y[j])


	X_train = list()
	y_train = list()

	# print(temp_X.shape, temp_y.shape)

	sum_list = [0]*10
	j = 0
	while sum(sum_list)!=5000:
		if sum_list[temp_y[j]]<500:
			# print(temp_y[j])
			# print(temp_X[j])
			sum_list[temp_y[j]]+=1
			X_train.append(temp_X[j])
			y_train.append(temp_y[j])
			
		j+=1

	X_p, y_p = shuffle(np.asarray(X_train), np.asarray(y_train))
	for i in range(5):
		print('Fold:', str(i+1))
		X_train, X_test = np.copy(np.concatenate((X_p[:1000*i], X_p[1000*(i+1):]), axis=0)), np.copy(X_p[1000*i:1000*(i+1)])
		y_train, y_test = np.copy(np.concatenate((y_p[:1000*i], y_p[1000*(i+1):]), axis=0)), np.copy(y_p[1000*i:1000*(i+1)])

		scaler = preprocessing.StandardScaler().fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		y_test = np.copy(y_test)

		assert X_train.shape[0]==4000
		assert X_test.shape[0]==1000

		# print(X_train)
		# print(y_train)
		print(X_train.shape, y_train.shape)

		clf = svm.SVC(kernel='linear', decision_function_shape='ovo')
		print('training...')
		clf.fit(X_train, y_train)		
		print('predicting...')
		preds = clf.predict(X_test)
		print("Fold: "+str(i+1))
		print("Accuracy:", accuracy_score(y_test, preds))
		print("F1 Score:", f1_score(y_test, preds, average=None))
		print("Confusion Matrix:", confusion_matrix(y_test, preds), '\n')
		with open('no_kernel_ovo_fold_'+str(i), 'wb') as f:
			pickle.dump(clf, f)


	for i in range(5):
		print('Fold:', str(i+1))
		X_train, X_test = np.copy(np.concatenate((X_p[:1000*i], X_p[1000*(i+1):]), axis=0)), np.copy(X_p[1000*i:1000*(i+1)])
		y_train, y_test = np.copy(np.concatenate((y_p[:1000*i], y_p[1000*(i+1):]), axis=0)), np.copy(y_p[1000*i:1000*(i+1)])

		scaler = preprocessing.StandardScaler().fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		y_test = np.copy(y_test)

		assert X_train.shape[0]==4000
		assert X_test.shape[0]==1000

		# print(X_train)
		# print(y_train)
		print(X_train.shape, y_train.shape)

		clf = svm.SVC(kernel='linear', decision_function_shape='ovr')
		print('training...')
		clf.fit(X_train, y_train)		
		print('predicting...')
		preds = clf.predict(X_test)
		print("Fold: "+str(i+1))
		print("Accuracy:", accuracy_score(y_test, preds))
		print("F1 Score:", f1_score(y_test, preds, average=None))
		print("Confusion Matrix:", confusion_matrix(y_test, preds), '\n')
		with open('no_kernel_ovr_fold_'+str(i), 'wb') as f:
			pickle.dump(clf, f)

def rbf_kernel_svm():
	from sklearn import svm
	from sklearn import preprocessing
	from sklearn.utils import shuffle
	from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
	import pickle

	X_, y_ = get_dataset()
	"""
	
	"""
	# print(X)
	# print(y)
	# print(X.shape)
	# print(y.shape)
	X, y = np.copy(X_), np.copy(y_)

	# for i in range(X.shape[0]):
	# print("Fold:", str(i+1))
	temp_X = None
	temp_y = None

	for j in range(X.shape[0]):
		# if j==i:
		# 	continue
		try:
			temp_X = np.concatenate((temp_X, X[j]), axis=0)
			temp_y = np.concatenate((temp_y, y[j]), axis=0)
		except:
			temp_X = np.copy(X[j])
			temp_y = np.copy(y[j])


	X_train = list()
	y_train = list()

	# print(temp_X.shape, temp_y.shape)

	sum_list = [0]*10
	j = 0
	while sum(sum_list)!=5000:
		if sum_list[temp_y[j]]<500:
			# print(temp_y[j])
			# print(temp_X[j])
			sum_list[temp_y[j]]+=1
			X_train.append(temp_X[j])
			y_train.append(temp_y[j])
			
		j+=1

	X_p, y_p = shuffle(np.asarray(X_train), np.asarray(y_train))
	for i in range(5):
		print('Fold:', str(i+1))
		X_train, X_test = np.copy(np.concatenate((X_p[:1000*i], X_p[1000*(i+1):]), axis=0)), np.copy(X_p[1000*i:1000*(i+1)])
		y_train, y_test = np.copy(np.concatenate((y_p[:1000*i], y_p[1000*(i+1):]), axis=0)), np.copy(y_p[1000*i:1000*(i+1)])

		scaler = preprocessing.StandardScaler().fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		y_test = np.copy(y_test)

		assert X_train.shape[0]==4000
		assert X_test.shape[0]==1000

		# print(X_train)
		# print(y_train)
		print(X_train.shape, y_train.shape)

		clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
		print('training...')
		clf.fit(X_train, y_train)		
		print('predicting...')
		preds = clf.predict(X_test)
		print("Fold: "+str(i+1))
		print("Accuracy:", accuracy_score(y_test, preds))
		print("F1 Score:", f1_score(y_test, preds, average=None))
		print("Confusion Matrix:", confusion_matrix(y_test, preds), '\n')
		with open('rbf_kernel_ovo_fold_'+str(i), 'wb') as f:
			pickle.dump(clf, f)


	for i in range(5):
		print('Fold:', str(i+1))
		X_train, X_test = np.copy(np.concatenate((X_p[:1000*i], X_p[1000*(i+1):]), axis=0)), np.copy(X_p[1000*i:1000*(i+1)])
		y_train, y_test = np.copy(np.concatenate((y_p[:1000*i], y_p[1000*(i+1):]), axis=0)), np.copy(y_p[1000*i:1000*(i+1)])

		scaler = preprocessing.StandardScaler().fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		y_test = np.copy(y_test)

		assert X_train.shape[0]==4000
		assert X_test.shape[0]==1000

		# print(X_train)
		# print(y_train)
		print(X_train.shape, y_train.shape)

		clf = svm.SVC(kernel='rbf', decision_function_shape='ovr')
		print('training...')
		clf.fit(X_train, y_train)		
		print('predicting...')
		preds = clf.predict(X_test)
		print("Fold: "+str(i+1))
		print("Accuracy:", accuracy_score(y_test, preds))
		print("F1 Score:", f1_score(y_test, preds, average=None))
		print("Confusion Matrix:", confusion_matrix(y_test, preds), '\n')
		with open('rbf_kernel_ovr_fold_'+str(i), 'wb') as f:
			pickle.dump(clf, f)

def quad_kernel_svm():
	from sklearn import svm
	from sklearn import preprocessing
	from sklearn.utils import shuffle
	from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
	import pickle

	X_, y_ = get_dataset()
	"""
	
	"""
	# print(X)
	# print(y)
	# print(X.shape)
	# print(y.shape)
	X, y = np.copy(X_), np.copy(y_)

	# for i in range(X.shape[0]):
	# print("Fold:", str(i+1))
	temp_X = None
	temp_y = None

	for j in range(X.shape[0]):
		# if j==i:
		# 	continue
		try:
			temp_X = np.concatenate((temp_X, X[j]), axis=0)
			temp_y = np.concatenate((temp_y, y[j]), axis=0)
		except:
			temp_X = np.copy(X[j])
			temp_y = np.copy(y[j])


	X_train = list()
	y_train = list()

	# print(temp_X.shape, temp_y.shape)

	sum_list = [0]*10
	j = 0
	while sum(sum_list)!=5000:
		if sum_list[temp_y[j]]<500:
			# print(temp_y[j])
			# print(temp_X[j])
			sum_list[temp_y[j]]+=1
			X_train.append(temp_X[j])
			y_train.append(temp_y[j])
			
		j+=1

	X_p, y_p = shuffle(np.asarray(X_train), np.asarray(y_train))
	for i in range(5):
		print('Fold:', str(i+1))
		X_train, X_test = np.copy(np.concatenate((X_p[:1000*i], X_p[1000*(i+1):]), axis=0)), np.copy(X_p[1000*i:1000*(i+1)])
		y_train, y_test = np.copy(np.concatenate((y_p[:1000*i], y_p[1000*(i+1):]), axis=0)), np.copy(y_p[1000*i:1000*(i+1)])

		scaler = preprocessing.StandardScaler().fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		y_test = np.copy(y_test)

		assert X_train.shape[0]==4000
		assert X_test.shape[0]==1000

		# print(X_train)
		# print(y_train)
		print(X_train.shape, y_train.shape)

		clf = svm.SVC(kernel='poly', decision_function_shape='ovo', degree=2, coef0=1)
		print('training...')
		clf.fit(X_train, y_train)		
		print('predicting...')
		preds = clf.predict(X_test)
		print("Fold: "+str(i+1))
		print("Accuracy:", accuracy_score(y_test, preds))
		print("F1 Score:", f1_score(y_test, preds, average=None))
		print("Confusion Matrix:", confusion_matrix(y_test, preds), '\n')
		with open('quad_kernel_ovo_fold_'+str(i), 'wb') as f:
			pickle.dump(clf, f)


	for i in range(5):
		print('Fold:', str(i+1))
		X_train, X_test = np.copy(np.concatenate((X_p[:1000*i], X_p[1000*(i+1):]), axis=0)), np.copy(X_p[1000*i:1000*(i+1)])
		y_train, y_test = np.copy(np.concatenate((y_p[:1000*i], y_p[1000*(i+1):]), axis=0)), np.copy(y_p[1000*i:1000*(i+1)])

		scaler = preprocessing.StandardScaler().fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		y_test = np.copy(y_test)

		assert X_train.shape[0]==4000
		assert X_test.shape[0]==1000

		# print(X_train)
		# print(y_train)
		print(X_train.shape, y_train.shape)

		clf = svm.SVC(kernel='poly', decision_function_shape='ovr', degree=2, coef0=1)
		print('training...')
		clf.fit(X_train, y_train)		
		print('predicting...')
		preds = clf.predict(X_test)
		print("Fold: "+str(i+1))
		print("Accuracy:", accuracy_score(y_test, preds))
		print("F1 Score:", f1_score(y_test, preds, average=None))
		print("Confusion Matrix:", confusion_matrix(y_test, preds), '\n')
		with open('quad_kernel_ovr_fold_'+str(i), 'wb') as f:
			pickle.dump(clf, f)

####################
#	Main
####################

if __name__ == '__main__':
	# no_kernel_svm()
	# rbf_kernel_svm()
	# quad_kernel_svm()