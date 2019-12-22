from helpers import *

parser = argparse.ArgumentParser(description='Image Detection')
parser.add_argument('-use_trained_model', action = 'store_true')
args = parser.parse_args()

# Load the data
[train_x, train_y, test_x, test_y, classes] = loadDataset()
print('\n\n\n')

# data is 64x64x3 = 12288x1
(n_x, m) = train_x.shape
# labels are 6x1 = 6
n_y = train_y.shape[0]


# to prevent overwriting of tensors
ops.reset_default_graph()

# Create variable placeholders
X = tf.placeholder(tf.float32, shape = [n_x, None], name = 'X')
Y = tf.placeholder(tf.float32, shape = [n_y, None], name = 'Y')

# Initialize Parameters with the xavier_initializer (bias as zero)
# We will have a three layer neural network
# with 25--12--6 layers
W1 = tf.get_variable("W1", [25, 12288], initializer = xavier_initializer(seed=1))
b1 = tf.get_variable("b1", [25, 1], initializer = tf.zeros_initializer())
W2 = tf.get_variable("W2", [12, 25], initializer = xavier_initializer(seed=1))
b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer())
W3 = tf.get_variable("W3", [6, 12], initializer = xavier_initializer(seed=1))
b3 = tf.get_variable("b3", [6, 1], initializer = tf.zeros_initializer())
# load parameters into a dict
parameters = {	"W1": W1,
				"b1": b1,
				"W2": W2,
				"b2": b2,
				"W3": W3,
				"b3": b3  }

if(not args.use_trained_model):
	# Forward propagation
	yhat = forward(X, parameters)

	# Define the loss
	cost = costFunction(yhat, Y)

	# Learning parameters
	learning_rate = 0.0001
	epochs = 1500
	mini_batch_size = 32
	num_batches = m / mini_batch_size

	# Create an adam optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

	# init global variables
	init = tf.global_variables_initializer()
	costs = []

	# start the session
	with tf.Session() as sess:
		sess.run(init)

		# Returns the accuracy at the current stage
		def accuracy():
			correct = tf.equal(tf.argmax(yhat), tf.argmax(Y))
			accuracy =  tf.reduce_mean(tf.cast(correct, 'float'))
			train_accuracy = accuracy.eval({X: train_x, Y: train_y})
			test_accuracy = accuracy.eval({X: test_x, Y: test_y})
			return (train_accuracy, test_accuracy)

		# Training
		for epoch in range(epochs):
			epoch_cost = 0

			for (batch_x, batch_y) in batchGenerator(train_x, train_y, mini_batch_size):
				_, iter_cost = sess.run([optimizer, cost], feed_dict = {X: batch_x, Y: batch_y})
				epoch_cost += iter_cost
			epoch_cost /= num_batches

			# Print the cost every 100th epoch
			if epoch % 100 == 0:
				print("\nCost after {}th epoch = {}".format(epoch, epoch_cost), flush = True)
				train_acc, test_acc = accuracy()
				print("Accuracy after {}th epoch :".format(epoch), flush = True)
				print("Train Accuracy = {}".format(train_acc), flush = True)
				print("Test Accuracy = {}\n".format(test_acc), flush = True)
			costs.append(epoch_cost)
			if epoch % 10 == 0:
				print("Running on the {}th epoch".format(epoch), flush = True)

		final_parameters = sess.run(parameters)
		print("Trained parameters successfully", flush = True)
		train_acc, test_acc = accuracy()
		print("Accuracy after Training :", flush = True)
		print("Train Accuracy = {}".format(train_acc), flush = True)
		print("Test Accuracy = {}".format(test_acc), flush = True)

		# create saver object
		saver = tf.train.Saver()
		# Save the model
		saver.save(sess, 'model/trained_model')
		
		# plot the cost
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()
else:
	# create saver object
	saver = tf.train.Saver()
	# import the graph from the file
	# imported_graph = tf.train.import_meta_graph('model/trained_model.meta')
	with tf.Session() as sess:
		saver.restore(sess, 'model/trained_model')		

		# Forward propagation
		yhat = forward(X, parameters)
		correct = tf.equal(tf.argmax(yhat), tf.argmax(Y))
		accuracy =  tf.reduce_mean(tf.cast(correct, 'float'))
		train_accuracy = accuracy.eval({X: train_x, Y: train_y})
		test_accuracy = accuracy.eval({X: test_x, Y: test_y})
	print("Train Accuracy = {}".format(train_accuracy), flush = True)
	print("Test Accuracy = {}".format(test_accuracy), flush = True)