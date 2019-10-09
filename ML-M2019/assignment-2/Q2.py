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


def make_ROC():
	from sklearn import svm
	from sklearn.metrics import roc_curve
	from sklearn.preprocessing import label_binarize
	from scipy import interp

	test_labels = label_binarize(test_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	n_classes = test_labels.shape[1]

	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
		print(test_labels[:, i])
		print(y_score[:, i])
		fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], y_score[:, i])

	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
	    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	mean_tpr /= n_classes

	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr

	plt.figure(figsize=(20, 20))
	for i in range(n_classes):
		plt.plot(fpr[i], tpr[i], label="ROC curve for class "+str(i))

	plt.plot(range(2), range(2), 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.2])

	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title("ROC Curves for classes")
	plt.legend(loc='lower right')
	plt.savefig("ROC_curves.png")
	plt.show()

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
		# with open('no_kernel_ovo_fold_'+str(i), 'wb') as f:
		# 	pickle.dump(clf, f)

		from sklearn import svm
		from sklearn.metrics import roc_curve
		from sklearn.preprocessing import label_binarize
		from scipy import interp

		test_labels = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
		n_classes = test_labels.shape[1]
		y_score = clf.decision_function(X_test)
		# Compute ROC curve and ROC area for each class
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for z in range(n_classes):
			# print(test_labels[:, z])
			# print(y_score[:, z])
			fpr[z], tpr[z], _ = roc_curve(test_labels[:, z], y_score[:, z])

		all_fpr = np.unique(np.concatenate([fpr[z] for z in range(n_classes)]))
		mean_tpr = np.zeros_like(all_fpr)
		for z in range(n_classes):
		    mean_tpr += interp(all_fpr, fpr[z], tpr[z])

		mean_tpr /= n_classes

		fpr["macro"] = all_fpr
		tpr["macro"] = mean_tpr

		plt.figure(figsize=(20, 20))
		for z in range(n_classes):
			plt.plot(fpr[z], tpr[z], label="ROC curve for class "+str(z))

		plt.plot(range(2), range(2), 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.2])

		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title("ROC Curves for classes")
		plt.legend(loc='lower right')
		plt.savefig("fold_"+str(i)+"_ROC_curves_ovo_linear.png")
		plt.show()


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
		# with open('no_kernel_ovr_fold_'+str(i), 'wb') as f:
		# 	pickle.dump(clf, f)


		from sklearn import svm
		from sklearn.metrics import roc_curve
		from sklearn.preprocessing import label_binarize
		from scipy import interp

		test_labels = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
		n_classes = test_labels.shape[1]
		y_score = clf.decision_function(X_test)
		# Compute ROC curve and ROC area for each class
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for z in range(n_classes):
			# print(test_labels[:, z])
			# print(y_score[:, z])
			fpr[z], tpr[z], _ = roc_curve(test_labels[:, z], y_score[:, z])

		all_fpr = np.unique(np.concatenate([fpr[z] for z in range(n_classes)]))
		mean_tpr = np.zeros_like(all_fpr)
		for z in range(n_classes):
		    mean_tpr += interp(all_fpr, fpr[z], tpr[z])

		mean_tpr /= n_classes

		fpr["macro"] = all_fpr
		tpr["macro"] = mean_tpr

		plt.figure(figsize=(20, 20))
		for z in range(n_classes):
			plt.plot(fpr[z], tpr[z], label="ROC curve for class "+str(z))

		plt.plot(range(2), range(2), 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.2])

		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title("ROC Curves for classes")
		plt.legend(loc='lower right')
		plt.savefig("fold_"+str(i)+"ROC_curves_ovr_linear.png")
		plt.show()


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
		# with open('rbf_kernel_ovo_fold_'+str(i), 'wb') as f:
		# 	pickle.dump(clf, f)

		from sklearn import svm
		from sklearn.metrics import roc_curve
		from sklearn.preprocessing import label_binarize
		from scipy import interp

		test_labels = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
		n_classes = test_labels.shape[1]
		y_score = clf.decision_function(X_test)
		# Compute ROC curve and ROC area for each class
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for z in range(n_classes):
			# print(test_labels[:, z])
			# print(y_score[:, z])
			fpr[z], tpr[z], _ = roc_curve(test_labels[:, z], y_score[:, z])

		all_fpr = np.unique(np.concatenate([fpr[z] for z in range(n_classes)]))
		mean_tpr = np.zeros_like(all_fpr)
		for z in range(n_classes):
		    mean_tpr += interp(all_fpr, fpr[z], tpr[z])

		mean_tpr /= n_classes

		fpr["macro"] = all_fpr
		tpr["macro"] = mean_tpr

		plt.figure(figsize=(20, 20))
		for z in range(n_classes):
			plt.plot(fpr[z], tpr[z], label="ROC curve for class "+str(z))

		plt.plot(range(2), range(2), 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.2])

		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title("ROC Curves for classes")
		plt.legend(loc='lower right')
		plt.savefig("fold_"+str(i)+"ROC_curves_ovo_rbf.png")
		plt.show()


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
		# with open('rbf_kernel_ovr_fold_'+str(i), 'wb') as f:
		# 	pickle.dump(clf, f)

		from sklearn import svm
		from sklearn.metrics import roc_curve
		from sklearn.preprocessing import label_binarize
		from scipy import interp

		test_labels = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
		n_classes = test_labels.shape[1]
		y_score = clf.decision_function(X_test)
		# Compute ROC curve and ROC area for each class
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for z in range(n_classes):
			# print(test_labels[:, z])
			# print(y_score[:, z])
			fpr[z], tpr[z], _ = roc_curve(test_labels[:, z], y_score[:, z])

		all_fpr = np.unique(np.concatenate([fpr[z] for z in range(n_classes)]))
		mean_tpr = np.zeros_like(all_fpr)
		for z in range(n_classes):
		    mean_tpr += interp(all_fpr, fpr[z], tpr[z])

		mean_tpr /= n_classes

		fpr["macro"] = all_fpr
		tpr["macro"] = mean_tpr

		plt.figure(figsize=(20, 20))
		for z in range(n_classes):
			plt.plot(fpr[z], tpr[z], label="ROC curve for class "+str(z))

		plt.plot(range(2), range(2), 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.2])

		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title("ROC Curves for classes")
		plt.legend(loc='lower right')
		plt.savefig("fold_"+str(i)+"ROC_curves_ovr_rbf.png")
		plt.show()

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
		# with open('quad_kernel_ovo_fold_'+str(i), 'wb') as f:
		# 	pickle.dump(clf, f)

		from sklearn import svm
		from sklearn.metrics import roc_curve
		from sklearn.preprocessing import label_binarize
		from scipy import interp

		test_labels = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
		n_classes = test_labels.shape[1]
		y_score = clf.decision_function(X_test)
		# Compute ROC curve and ROC area for each class
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for z in range(n_classes):
			# print(test_labels[:, z])
			# print(y_score[:, z])
			fpr[z], tpr[z], _ = roc_curve(test_labels[:, z], y_score[:, z])

		all_fpr = np.unique(np.concatenate([fpr[z] for z in range(n_classes)]))
		mean_tpr = np.zeros_like(all_fpr)
		for z in range(n_classes):
		    mean_tpr += interp(all_fpr, fpr[z], tpr[z])

		mean_tpr /= n_classes

		fpr["macro"] = all_fpr
		tpr["macro"] = mean_tpr

		plt.figure(figsize=(20, 20))
		for z in range(n_classes):
			plt.plot(fpr[z], tpr[z], label="ROC curve for class "+str(z))

		plt.plot(range(2), range(2), 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.2])

		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title("ROC Curves for classes")
		plt.legend(loc='lower right')
		plt.savefig("fold_"+str(i)+"ROC_curves_ovo_quad.png")
		plt.show()


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
		# with open('quad_kernel_ovr_fold_'+str(i), 'wb') as f:
		# 	pickle.dump(clf, f)

		from sklearn import svm
		from sklearn.metrics import roc_curve
		from sklearn.preprocessing import label_binarize
		from scipy import interp

		test_labels = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
		n_classes = test_labels.shape[1]
		y_score = clf.decision_function(X_test)
		# Compute ROC curve and ROC area for each class
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for z in range(n_classes):
			# print(test_labels[:, z])
			# print(y_score[:, z])
			fpr[z], tpr[z], _ = roc_curve(test_labels[:, z], y_score[:, z])

		all_fpr = np.unique(np.concatenate([fpr[z] for z in range(n_classes)]))
		mean_tpr = np.zeros_like(all_fpr)
		for z in range(n_classes):
		    mean_tpr += interp(all_fpr, fpr[z], tpr[z])

		mean_tpr /= n_classes

		fpr["macro"] = all_fpr
		tpr["macro"] = mean_tpr

		plt.figure(figsize=(20, 20))
		for z in range(n_classes):
			plt.plot(fpr[z], tpr[z], label="ROC curve for class "+str(z))

		plt.plot(range(2), range(2), 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.2])

		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title("ROC Curves for classes")
		plt.legend(loc='lower right')
		plt.savefig("fold_"+str(i)+"ROC_curves_ovr_quad.png")
		plt.show()

####################
#	Main
####################

if __name__ == '__main__':
	no_kernel_svm()
	rbf_kernel_svm()
	quad_kernel_svm()