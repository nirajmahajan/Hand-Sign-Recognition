import logging
logging.getLogger('tensorflow').disabled = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import math
import h5py
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.nn import relu
from tensorflow.contrib.layers import xavier_initializer, flatten, fully_connected
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

np.random.seed(1)
tf.set_random_seed(1)

# returns the train dataset, test dataset and the classes
def loadDataset():
	np.random.seed(1)
	tf.set_random_seed(1)
	filename = 'data/train_signs.h5'
	if(not os.path.isfile(filename)):
		print('{} not found'.format(filename))
		os._exit(1)

	with h5py.File(filename, 'r') as f:
		# Get the data - flatten and normalize it
		data_train_x = np.array(f['train_set_x'])/255.

		# Convert the labels to one_hot
		data_train_y = np.array(f['train_set_y'])
		data_train_y = to_one_hot(data_train_y, 6).T

	filename = 'data/test_signs.h5'
	if(not os.path.isfile(filename)):
		print('{} not found'.format(filename))
		os._exit(1)

	with h5py.File(filename, 'r') as f:
		# Get the data - flatten and normalize it
		data_test_x = np.array(f['test_set_x'])/255.

		# Convert the labels to one_hot
		data_test_y = np.array(f['test_set_y'])
		data_test_y = to_one_hot(data_test_y, 6).T

		# get the classes
		classes = np.array(f["list_classes"])

	return [data_train_x, data_train_y, data_test_x, data_test_y, classes]

# Displays an image from the dataset
def displayImage(img):
	plt.imshow(img.reshape(64,64,3))
	plt.show()

# generates random mini_batches
def batchGenerator(X, Y, mini_batch_size):
	np.random.seed(1)
	tf.set_random_seed(1)
	m = X.shape[0]
	mini_batches = []
	
	# Step 1: Shuffle (X, Y)
	permutation = list(np.random.permutation(m))
	shuffled_X = X[permutation,:,:,:]
	shuffled_Y = Y[permutation,:]

	# Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
	num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
		mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	# Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
		mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	return mini_batches
	
# Returns the one_hot representation of input labels
def to_one_hot(labels, num_classes):
	np.random.seed(1)
	tf.set_random_seed(1)
	num_classes = tf.constant(num_classes, name='num_classes')
	one_hot_matrix = tf.one_hot(indices=labels, depth=num_classes, axis=0)
	
	with tf.Session() as sess:
		one_hot = sess.run(one_hot_matrix)
	
	return one_hot

# forward propagation function
def forward(X, parameters):
	np.random.seed(1)
	tf.set_random_seed(1)
	# Get the parameters from the dict
	W1 = parameters['W1']
	W2 = parameters['W2']
	
	# CONV 1
	Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
	A1 = tf.nn.relu(Z1)
	# MAXPOOL 1
	P1 = tf.nn.max_pool(A1, ksize = [1, 8, 8, 1], strides = [1, 8, 8, 1], padding='SAME')
	# CONV 2
	Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
	A2 = tf.nn.relu(Z2)
	# MAXPOOL 2
	P2 = tf.nn.max_pool(A2, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding='SAME')
	# FLATTEN
	P = flatten(P2)
	# FULLY-CONNECTED
	Z3 = fully_connected(P, 6, activation_fn=None)

	return Z3

# Cost function (sigmoid cross entropy loss)
def costFunction(yhat, y):
	np.random.seed(1)
	tf.set_random_seed(1)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yhat, labels=y))
	return cost

def predictor(X, parameters):
	np.random.seed(1)
	tf.set_random_seed(1)
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	W3 = parameters["W3"]
	b3 = parameters["b3"]
	
	params = {  "W1": W1,
				"b1": b1,
				"W2": W2,
				"b2": b2,
				"W3": W3,
				"b3": b3  }
	
	x = tf.placeholder("float", [12288, 1])
	
	z3 = forward(x, params)
	p = tf.argmax(z3)
	prob = tf.reduce_max(z3)

	sess = tf.Session()
	[prediction, prob] = sess.run([p, prob], feed_dict = {x: X})
	sess.close()
	return [prediction, prob]