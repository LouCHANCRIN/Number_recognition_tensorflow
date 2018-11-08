import pandas as pd
import numpy as np
#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import struct as st


class data:

    def __init__(self, train_image, train_label, test_image, test_label, split):
        train_data, train_label = self.unpack_image(train_image, train_label)
        self.test_data, self.test_label = self.unpack_image(test_image, test_label)
        train_length = int(np.shape(train_data)[0] * split)
        self.train_data = train_data[:train_length]
        self.train_label = train_label[:train_length]
        self.validation_data = train_data[train_length:]
        self.validation_label = train_label[train_length:]

    def unpack_image(self, image_path, label_path):
        filename = {'images' : image_path, 'labels' : label_path}
        image = open(filename['images'], 'rb')
        magic = st.unpack('>I', image.read(4))
        nb_image = st.unpack('>I', image.read(4))[0]
        nb_line = st.unpack('>I', image.read(4))[0]
        nb_col = st.unpack('>I', image.read(4))[0]
        tot_size = nb_image * nb_line * nb_col
        ret_image = (255.0 - np.asarray(st.unpack('>' + 'B' *
            tot_size, image.read(tot_size))).reshape((nb_image, nb_line, nb_col)))
        ret_image = np.float32(ret_image)
        label = open(filename['labels'], 'rb')
        magic = st.unpack('>I', label.read(4))
        nb_image = st.unpack('>I', label.read(4))[0]
        ret_label = np.asarray(st.unpack('>' + 'B' * nb_image, label.read(nb_image)))
        return (ret_image / 255, ret_label)
