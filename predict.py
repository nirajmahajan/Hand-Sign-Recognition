import scipy
from PIL import Image
from scipy import ndimage
from helpers import *

parser = argparse.ArgumentParser(description='Sign Detection')
parser.add_argument('--path', required = True)

args = vars(parser.parse_args())

if(not os.path.exists(args['path'])):
	print('Not a valid path')
	os._exit(1)

if(not os.path.isfile(args['path'])):
	print('Not a file')
	os._exit(1)

# to prevent overwriting of tensors
ops.reset_default_graph()

fname = args['path']
image = np.array(ndimage.imread(fname, flatten=False))
image = image/255.
image = scipy.misc.imresize(image, size=(64,64))
image = image[:, :, 0:3]
my_image = image.reshape((1, 64*64*3)).T

n_x = 12288
n_y = 6

plt.imshow(image)
plt.title('Resized image')

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

# create saver object
saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess, 'model/trained_model')		

	# Forward propagation
	yhat = tf.nn.softmax(forward(X, parameters))
	p = tf.argmax(yhat)
	prob = tf.reduce_max(yhat)

	[pred, prob] = sess.run([p, prob], feed_dict = {X: my_image})

	print('\n\n\n')
	print("Prediction: {}".format(pred))
	print("Probability: {}".format(prob))

plt.show()