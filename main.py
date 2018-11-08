import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import struct as st
import get_image as get
import matplotlib.pyplot as plt

ops.reset_default_graph()

def main(data):
# MODEL PARAMETERS
    nb_image, line, col, nb_chanel = np.shape(data.train_data)
    alpha = 0.5
    num_iters = 2000
    batch_size = 64
    nb_class = 10

if __name__ == '__main__':
    train_im = 'mnist/train_image'
    train_lab = 'mnist/train_label'
    test_im = 'mnist/t10_image'
    test_lab = 'mnist/t10_label'
    split = 0.8 # % of data to train, rest is validation data
    data = get.data(train_im, train_lab, test_im, test_lab, split)
    length, line, col = np.shape(data.train_data)
    data.train_data = np.reshape(data.train_data, (length, line, col, 1))
    main(data)
