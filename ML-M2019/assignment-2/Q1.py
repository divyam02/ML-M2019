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
	plt.scatter(x_0, y_0, color="red", edgecolors="white")
	plt.scatter(x_1, y_1, color="blue", edgecolors="white")
	plt.savefig('Q1_a')

def q1_b():
	pass

def q1_c():
	pass

def q1_d():
	pass


####################	
#	Main	
####################

if __name__ == '__main__':
	# Uncomment question to run.
	# extract_h5_files()
	q1_a()
	# q1_b()
	# q1_c()
	# q1_d()