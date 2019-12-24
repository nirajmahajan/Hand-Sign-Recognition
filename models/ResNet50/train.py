from helpers import *
from keras.utils import plot_model
np.random.seed(1)

parser = argparse.ArgumentParser(description='Image Detection')
parser.add_argument('-use_trained_model', action = 'store_true')
args = parser.parse_args()

[train_x, train_y, test_x, test_y, classes] = loadDataset()
print('Loaded the dataset', flush = True)

if(not args.use_trained_model):
	myModel = MyModel(train_x.shape[1:])
	print('Created the model', flush = True)
	myModel.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	print('Compiled the model', flush = True)
	myModel.fit(x = train_x, y = train_y, epochs = 25, batch_size = 32)

	preds = myModel.evaluate(test_x, test_y, verbose=1)
	print("\nResults after training:\nLoss = {}".format(preds[0]))
	print("Test Accuracy = {}".format(preds[1]))
	myModel.save('model/trained_model.h5.big')
	plot_model(myModel, to_file='model/trained_model.png')
else:
	print('Loading the model', flush = True)
	myModel = load_model('model/trained_model.h5.big')
	print('Model Loaded Successfully\n', flush = True)
	preds = myModel.evaluate(test_x, test_y, verbose=1)
	preds1 = myModel.evaluate(train_x, train_y, verbose=1)
	print("\nTrain Accuracy = {}".format(preds1[1]))
	print("Test Accuracy = {}".format(preds[1]))