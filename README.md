# Number recognition tensorflow

## About number recognition :

* Number Recognition goal is to recognize handwritten digit.

* In this project I am using [tensorflow](https://www.tensorflow.org/).

## About the code :

* I am using a [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network).

* Data is splitted in 3 : 48000 images for training set, 12000 for validation set and 10000 for test set.

* Using [softmax](https://en.wikipedia.org/wiki/Softmax_function) as output layer.

* Using [cross entropy](https://en.wikipedia.org/wiki/Cross_entropy) as cost function.

* Using [AdamOptimizer](http://ruder.io/optimizing-gradient-descent/index.html#adam).

## Data :

* 70000 images of handwritten digit with a format of 28x28x1 ([MNIST data](http://yann.lecun.com/exdb/mnist/)).

* Data is splitted in 3 : 48000 images for training set, 12000 for validation set and 10000 for test set.

## Requirements :

* Python3.7

* Tensorflow

* Numpy

* Matplotlib

## Usage :

* python3.7 main.py

## To do :

* Data augmentation

* Save the model

* Random batch

* More metrics

* Script to draw and test my model on my own handwritten digit
