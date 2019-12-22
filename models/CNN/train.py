from helpers import *

parser = argparse.ArgumentParser(description='Image Detection')
parser.add_argument('-use_trained_model', action = 'store_true')
args = parser.parse_args()

np.random.seed(1)
tf.set_random_seed(1)

# to prevent overwriting of tensors
ops.reset_default_graph()

# Load the data
[train_x, train_y, test_x, test_y, classes] = loadDataset()
print('\n\n\n')

conv_layers = {}
# data is 64x64x3 = 12288x1
# (1080x64x64x3)
(m, n_H0, n_W0, n_C0) = train_x.shape             
# labels are 6x1 = 6
# (1080, 6)
n_y = train_y.shape[1] 

if(not args.use_trained_model):
	# Create variable placeholders
	X = tf.placeholder(tf.float32, shape = [None, n_H0, n_W0, n_C0])
	Y = tf.placeholder(tf.float32, shape = [None, n_y])

	# Initialize Parameters with the xavier_initializer (bias as zero)
	# We will have a three layer neural network
	# with 25--12--6 layers
	W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer = xavier_initializer(seed=1))
	W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer = xavier_initializer(seed=1))
	# load parameters into a dict
	parameters = {	"W1": W1,
					"W2": W2  }

	acc_temp = 0

	# Forward propagation
	yhat = forward(X, parameters)

	# Define the loss
	cost = costFunction(yhat, Y)

	# Learning parameters
	learning_rate = 0.009
	epochs = 75
	mini_batch_size = 64
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
			correct = tf.equal(tf.argmax(yhat, 1), tf.argmax(Y, 1))
			accuracy =  tf.reduce_mean(tf.cast(correct, 'float'))
			train_accuracy = accuracy.eval({X: train_x, Y: train_y})
			test_accuracy = accuracy.eval({X: test_x, Y: test_y})
			return (train_accuracy, test_accuracy)

		# Training
		for epoch in range(epochs):
			epoch_cost = 0.

			for (batch_x, batch_y) in batchGenerator(train_x, train_y, mini_batch_size):
				_, iter_cost = sess.run([optimizer, cost], feed_dict = {X: batch_x, Y: batch_y})
				epoch_cost += iter_cost
			epoch_cost /= num_batches

			if epoch % 10 == 0:
				print("Running on the {}th epoch".format(epoch), flush = True)
			# Print the cost every 10th epoch
			if epoch % 5 == 0:
				print("\nCost after {}th epoch = {}".format(epoch, epoch_cost), flush = True)
				train_acc, test_acc = accuracy()
				acc_temp = test_acc
				print("Accuracy after {}th epoch :".format(epoch), flush = True)
				print("Train Accuracy = {}".format(train_acc), flush = True)
				print("Test Accuracy = {}\n".format(test_acc), flush = True)
			costs.append(epoch_cost)
			

		final_parameters = sess.run(parameters)
		print("Trained parameters successfully", flush = True)
		train_acc, test_acc = accuracy()
		print("Accuracy after Training :", flush = True)
		print("Train Accuracy = {}".format(train_acc), flush = True)
		print("Test Accuracy = {}".format(test_acc), flush = True)

		# create saver object
		saver = tf.train.Saver()
		# Save the model
		saver.save(sess, 'model/trained_model.ckpt')
		
		# plot the cost
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()
else:
	# create saver object
	# saver = tf.train.Saver()
	
	# import the graph from the file
	# imported_graph = 
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph('model/trained_model.ckpt.meta')
		# Forward propagation
		yhat = tf.get_default_graph().get_tensor_by_name("fully_connected/BiasAdd:0")
		X = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
		Y = tf.get_default_graph().get_tensor_by_name("Placeholder_1:0")
		W1 = tf.get_default_graph().get_tensor_by_name("W1:0")
		W2 = tf.get_default_graph().get_tensor_by_name("W2:0")

		saver.restore(sess, 'model/trained_model.ckpt')
		correct = tf.equal(tf.argmax(yhat, 1), tf.argmax(Y, 1))
		accuracy =  tf.reduce_mean(tf.cast(correct, 'float'))
		train_accuracy = accuracy.eval({X: train_x, Y: train_y})
		test_accuracy = accuracy.eval({X: test_x, Y: test_y})
	print("Train Accuracy = {}".format(train_accuracy), flush = True)
	print("Test Accuracy = {}".format(test_accuracy), flush = True)