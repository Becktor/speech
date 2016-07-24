import tensorflow as tf
import scipy
import scipy.io
import numpy as np
import batch
import time
#print 'started'
trainSet = scipy.io.loadmat('TrainBatch2.mat')['arr']
testSet = scipy.io.loadmat('TestBatch2.mat')['arr']
#print 'data loaded'
train = batch.Batch(trainSet)
#print 'train batched'
test = batch.Batch(testSet)
#print 'data batched'
# Import MINST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#train = mnist.train
#test = mnist.test
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

start_time = time.time()



# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 128
display_step = 10
dropout=0.75

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_hidden_3 = 256 # 3rd layer number of features
n_hidden_4 = 256 # 4th layer number of features
n_hidden_5 = 256 # 5th layer number of features

n_input = 4018   # Dataset data input (img shape: 98*41)
n_classes = 6    # Dataset total classes (6 emotions)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, dropout)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    #layer_2 = tf.nn.dropout(layer_2, dropout)
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    #layer_3 = tf.nn.dropout(layer_3, dropout)
    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    #layer_4 = tf.nn.dropout(layer_4, dropout)
    # Hidden layer with RELU activation
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.relu(layer_5)
    #layer_5 = tf.nn.dropout(layer_5, dropout)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
    #out_layer = tf.nn.dropout(out_layer, 0.5)
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
    'out': tf.Variable(tf.random_normal([n_hidden_5, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'b5': tf.Variable(tf.random_normal([n_hidden_5])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# Initializing the variables
init = tf.initialize_all_variables()
# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost)
            print "Accuracy:", accuracy.eval({x: train.images, y: train.labels})
            #print "Accuracy:", accuracy.eval({x: train.spectrograms, y: train.labels})
    print "Optimization Finished!"
    print "Accuracy:", accuracy.eval({x: test.images, y: test.labels})
    print "Accuracy:", accuracy.eval({x: train.images, y: train.labels})
    #print "Accuracy:", accuracy.eval({x: test.spectrograms, y: test.labels})
    print("--- %s seconds ---" % (time.time() - start_time))
