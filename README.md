# Hand-Sign Identifier

A hand sign classifier in Tensorflow (v1.15) using several models.

1. Neural Network : Accuracy of 77.50%
2. Convolutional Neural Network : Accuracy of 88.33 %

## Code Structure:

The repository has the following files:

- [helpers.py](https://github.com/nirajmahajan/Hand-Sign-Recognition/blob/master/helpers.py) : Contains all essential imports and functions
- [train.py](https://github.com/nirajmahajan/Hand-Sign-Recognition/blob/master/train.py) : Used for training and testing the accuracy of a model
- [predict.py](https://github.com/nirajmahajan/Hand-Sign-Recognition/blob/master/predict.py) : Used for predicting the digit in a given image
- [model/](https://github.com/nirajmahajan/Hand-Sign-Recognition/tree/master/model) : Has a pre-trained model
- [sample/](https://github.com/nirajmahajan/Hand-Sign-Recognition/tree/master/sample) : Has a few sample images
- [data/](https://github.com/nirajmahajan/Hand-Sign-Recognition/tree/master/data) : Contains the datasets in .h5 format

## Code Structure:

The code structure for all the models is quite similar. Each model has the following files:

- **helpers.py** : Contains all essential imports and functions
- **train.py** : Used for training and testing the accuracy of a model
- **predict.py** : Used for predicting the digit in a given image
- **model/** : Has a pre-trained model

Apart from this, the MNIST data and a few sample images are located in the [Utils](https://github.com/nirajmahajan/Digit-Recognition/tree/master/utils) folder.

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
