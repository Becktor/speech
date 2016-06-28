import tensorflow as tf
import scipy
import scipy.io
import numpy as np
import batch
#print 'started'
trainSet = scipy.io.loadmat('TrainBatch.mat')['arr']
testSet = scipy.io.loadmat('TestBatch.mat')['arr']
#print 'data loaded'
train = batch.Batch(trainSet)
#print 'train batched'
test = batch.Batch(testSet)
#print 'data batched'

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 4018])
    W = tf.Variable(tf.zeros([4018, 6]))
    b = tf.Variable(tf.zeros([6]))
    y_ = tf.placeholder(tf.float32, [None, 6])
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Multilayer Convolutional Network
    # First Layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 98, 41, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    # Second Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Third Layer
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    # Fourth Layer
    W_conv4 = weight_variable([5, 5, 128, 256])
    b_conv4 = bias_variable([256])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)



    # Fully Connected Layer
    W_fc1 = weight_variable([28 * 30 * 64, 2048])
    b_fc1 = bias_variable([2048])

    h_pool4_flat = tf.reshape(h_pool4, [-1, 28*30*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([2048, 6])
    b_fc2 = bias_variable([6])
    #print 'here'
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logit, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    ####print "y_ shape:", y_conv.get_shape()#print accuracy

    sess.run(tf.initialize_all_variables())

    for i in range(5000):
        batch = train.next_batch(10)

        #print batch[0].shape
        #print batch[1].shape

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print('testing now')
    suma=0.
    cntr=0.
    for i in range(400):
        tbatch = test.next_batch(10)
        cntr +=1
        suma += accuracy.eval(feed_dict={x: tbatch[0], y_: tbatch[1], keep_prob: 1.0})
    val = suma/cntr
    print suma
    print
    print("test accuracy %g" % (val))
