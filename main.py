import pandas as pd
import numpy as np
import tensorflow as tf
import get_image as get
import model as mod
import matplotlib.pyplot as plt

train_im = 'mnist/train_image'
train_lab = 'mnist/train_label'
test_im = 'mnist/t10_image'
test_lab = 'mnist/t10_label'
split = 0.8 # % of data to train, rest is validation data
data = get.data(train_im, train_lab, test_im, test_lab, split)

nb_image, line, col = np.shape(data.train_data)
data.train_data = np.reshape(data.train_data, [nb_image, line, col, 1])
a = np.shape(data.test_data)[0]
data.test_data = np.reshape(data.test_data, [a, line, col, 1])
a = np.shape(data.validation_data)[0]
data.validation_data = np.reshape(data.validation_data, [a, line, col, 1])

alpha = 0.001
nb_chanel = 1
nb_class = 10
batch_size = 128

x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, 10])

weight = {'conv1': tf.Variable(tf.random_normal([3,3,1,32])),
          'conv2': tf.Variable(tf.random_normal([3,3,32,64])),
          'conv3': tf.Variable(tf.random_normal([3,3,64,128])),
          'fc': tf.Variable(tf.random_normal([4*4*128,128])),
          'output': tf.Variable(tf.random_normal([128, nb_class]))}

bias = {'conv1': tf.Variable(tf.random_normal([32])),
        'conv2': tf.Variable(tf.random_normal([64])),
        'conv3': tf.Variable(tf.random_normal([128])),
        'fc': tf.Variable(tf.random_normal([128])),
        'output': tf.Variable(tf.random_normal([nb_class]))}

pred = mod.model(x, weight, bias)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    for e in range(0, 100):
        #for i in range(0, int(nb_image / batch_size)):
        for i in range(0, 100):
            batch_x = data.train_data[i*batch_size:min((i+1)*batch_size,len(data.train_data))]
            batch_y = data.train_label[i*batch_size:min((i+1)*batch_size,len(data.train_label))]

            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
        print("Epoch :", e, ", loss =", loss, ", acc =", acc)
        test_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x: data.test_data, y: data.test_label})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)

plt.plot(range(len(train_loss)), train_loss, 'b', label='Train loss')
plt.plot(range(len(test_loss)), test_loss, 'r', label='Test loss')
plt.title('Training and test loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.plot(range(len(train_accuracy)), train_loss, 'b', label='Train accuracy')
plt.plot(range(len(test_accuracy)), test_accuracy, 'r', label='Test accuracy')
plt.title('Training and test accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
