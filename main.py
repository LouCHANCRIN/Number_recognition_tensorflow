import numpy as np
import tensorflow as tf
import get_image as get
import model as mod
import matplotlib.pyplot as plt

train_im = 'mnist/train_image'
train_lab = 'mnist/train_label'
test_im = 'mnist/t10_image'
test_lab = 'mnist/t10_label'
split = 0.8 #(0.8) % of data to train, rest is validation data
data = get.data(train_im, train_lab, test_im, test_lab, split)

def plot(train_loss, test_loss, train_accuracy, test_accuracy):
    plt.plot(range(len(train_loss)), train_loss, 'b', label='Train loss')
    plt.plot(range(len(test_loss)), test_loss, 'r', label='Test loss')
    plt.title('Training(blue) and test(red) loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    plt.plot(range(len(train_accuracy)), train_accuracy, 'b', label='Train accuracy')
    plt.plot(range(len(test_accuracy)), test_accuracy, 'r', label='Test accuracy')
    plt.title('Training(blue) and test(red) accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

def my_weight_and_bias(nb_class):
    w = {'conv1': tf.Variable(tf.random_normal([3,3,1,32], seed=3)), #3
         'conv2': tf.Variable(tf.random_normal([3,3,32,64], seed=1)), #1
         'conv3': tf.Variable(tf.random_normal([3,3,64,128], seed=4)), #4
         'fc1': tf.Variable(tf.random_normal([4*4*128,128], seed=5)), #5
         'fc2': tf.Variable(tf.random_normal([128,128], seed=5)), #5
         'output': tf.Variable(tf.random_normal([128, nb_class], seed=4))} #4
    b = {'conv1': tf.Variable(tf.random_normal([32], seed=3)), #3
         'conv2': tf.Variable(tf.random_normal([64], seed=2)), #2
         'conv3': tf.Variable(tf.random_normal([128], seed=2)), #2
         'fc1': tf.Variable(tf.random_normal([128], seed=9)), #9
         'fc2': tf.Variable(tf.random_normal([128], seed=9)), #9
         'output': tf.Variable(tf.random_normal([nb_class], seed=5))} #5
    return (w, b)

def LeNet5_weight_and_bias(nb_class):
    w = {'conv1': tf.Variable(tf.random_normal([3,3,1,32], seed=3)), #3
         'conv2': tf.Variable(tf.random_normal([3,3,32,64], seed=1)), #1
         'fc1': tf.Variable(tf.random_normal([7*7*64,64], seed=5)), #5
         'fc2': tf.Variable(tf.random_normal([64,64], seed=5)), #5
         'output': tf.Variable(tf.random_normal([64, nb_class], seed=4))} #4
    b = {'conv1': tf.Variable(tf.random_normal([32], seed=3)), #3
         'conv2': tf.Variable(tf.random_normal([64], seed=2)), #2
         'fc1': tf.Variable(tf.random_normal([64], seed=9)), #9
         'fc2': tf.Variable(tf.random_normal([64], seed=9)), #9
         'output': tf.Variable(tf.random_normal([nb_class], seed=5))} #5
    return (w, b)

def main():
    alpha = 0.004 #0.004
    alpha = 0.005 #0.008
    num_epoch = 50 #23
    nb_chanel = 1
    nb_class = 10
    batch_size = 128
    nb_image, line, col = np.shape(data.train_data)
    data.train_data = np.reshape(data.train_data, [nb_image, line, col, 1])
    data.test_data = np.reshape(data.test_data, [np.shape(data.test_data)[0], line, col, 1])
    data.validation_data = np.reshape(data.validation_data,
            [np.shape(data.validation_data)[0], line, col, 1])
    x = tf.placeholder("float", [None, 28, 28, 1])
    y = tf.placeholder("float", [None, 10])

    #weight, bias = my_weight_and_bias(nb_class) 
    weight, bias = LeNet5_weight_and_bias(nb_class) 

    pred = mod.LeNet5(x, weight, bias)
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
        for e in range(0, num_epoch):
            for i in range(0, int(nb_image / batch_size)):
                batch_x = data.train_data[i*batch_size:min((i+1)*batch_size,
                    len(data.train_data))]
                batch_y = data.train_label[i*batch_size:min((i+1)*batch_size,
                    len(data.train_label))]
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            TmpLoss, TmpAcc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            TmpTestLoss, TmpTestAcc = sess.run([cost, accuracy],
                    feed_dict={x: data.test_data, y: data.test_label})
            print("Epoch :", e, ", Loss :", TmpLoss, ", Train acc :",
                    TmpAcc, ", Test acc :", TmpTestAcc)
            train_loss.append(TmpLoss)
            test_loss.append(TmpTestLoss)
            train_accuracy.append(TmpAcc)
            test_accuracy.append(TmpTestAcc)
        Val_Loss, Val_Acc = sess.run([cost, accuracy],
                feed_dict={x: data.validation_data, y: data.validation_label})
        print("Validation loss :", Val_Loss)
        print("Validation acc  :", Val_Acc * 100)
    plot(train_loss, test_loss, train_accuracy, test_accuracy)
    x = 0
    y = 0
    for i in range(0, num_epoch):
        if (test_loss[i] > x):
            x = test_accuracy[i]
            y = i
    print("max :", x, "(", y, ")")

if __name__ == "__main__":
    main()
