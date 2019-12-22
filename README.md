# Hand-Sign Identifier

A hand sign classifier in Tensorflow (v1.15) using several models.

1. [Neural Network](https://github.com/nirajmahajan/Hand-Sign-Recognition/tree/master/models/NN) : Accuracy of 77.50%
2. [Convolutional Neural Network](https://github.com/nirajmahajan/Hand-Sign-Recognition/tree/master/models/CNN) : Accuracy of 88.33 %

## Code Structure:

The code structure for all the models is quite similar. Each model has the following files:

- **helpers.py** : Contains all essential imports and functions
- **train.py** : Used for training and testing the accuracy of a model
- **predict.py** : Used for predicting the digit in a given image
- **model/** : Has a pre-trained model

Apart from this, the SIGNS data and a few sample images are located in the [Utils](https://github.com/nirajmahajan/Hand-Sign-Recognition/tree/master/utils) folder.

## Usage of Code:

1. To train any model:

   ```bash
   python3 train.py # Note that this will replace the pre existing model
   ```

2. To check the accuracy of any model (present in the 'model' directory):

   ```bash
   python3 train.py -use_trained_model
   ```

3. To predict the 'hand sign' from an image:

   ```bash
   python3 predict.py --path <Path To Image
   ```
