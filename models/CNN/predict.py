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
my_image = image.reshape((1, 64, 64, 3))

n_x = 12288
n_y = 6

plt.imshow(image)
plt.title('Resized image')

# create saver object
# saver = tf.train.Saver()

with tf.Session() as sess:
	saver = tf.train.import_meta_graph('model/trained_model.ckpt.meta')
	# Forward propagation
	yhat = tf.get_default_graph().get_tensor_by_name("fully_connected/BiasAdd:0")
	X = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
	Y = tf.get_default_graph().get_tensor_by_name("Placeholder_1:0")
	W1 = tf.get_default_graph().get_tensor_by_name("W1:0")
	W2 = tf.get_default_graph().get_tensor_by_name("W2:0")
	parameters = {	"W1": W1,
					"W2": W2  }

	saver.restore(sess, 'model/trained_model.ckpt')

	# Forward propagation
	yhat = tf.nn.softmax(tf.reshape(yhat, [-1]))

	# Load data

	p = tf.argmax(yhat)
	prob = tf.reduce_max(yhat)

	[pred, prob] = sess.run([p, prob], feed_dict = {X: my_image})
	yhat = sess.run(yhat, feed_dict = {X: my_image})

	print('\n\n\n')
	print("Prediction: {}".format(pred))
	# print("Probability: {}".format(prob))

plt.show()