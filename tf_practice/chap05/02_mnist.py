import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pdb

data_dir = "/home/mxd/software/data/MNIST"
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class == None:
	layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
	return tf.matmul(layer1, weights2) + biases2
    else:
	layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) 
	         + avg_class.average(biases1))
	return (tf.matmul(layer1, avg_class.average(weights2)) +  
	       avg_class.average(biases2))

def train(mnist):
    # input and output placeholder
    x = tf.placeholder(shape = [None, INPUT_NODE], dtype = tf.float32, 
                       name = "x-input")
    y_ = tf.placeholder(shape = [None, OUTPUT_NODE], dtype = tf.float32, 
                        name = "y-input")

    # net params and results with no moving average operation
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev = 0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape = [LAYER1_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev = 0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape = [OUTPUT_NODE]))
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # result with moving average operation. but is this result used?
    global_step = tf.Variable(0, trainable = False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, 
                    labels = tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 
                    mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
                 global_step = global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name = "train")

    # compute both the accuracy with and without average_op
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    correct_prediction_avg = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy_avg = tf.reduce_mean(tf.cast(correct_prediction_avg, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
	validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
	test_feed = {x: mnist.test.images, y_: mnist.test.labels}

	for i in range(TRAINING_STEPS):
	    xs, ys = mnist.train.next_batch(BATCH_SIZE)
	    sess.run(train_op, feed_dict = {x: xs, y_: ys})
	    if i % 1000 == 0:
		validate_acc = sess.run(accuracy_avg, feed_dict = validate_feed)
		print("After %d training step, validation accuracy with avg is %g." %
		      (i, validate_acc))
		test_acc = sess.run(accuracy_avg, feed_dict = test_feed)
		print("After %d training step, test accuracy with avg is %g." %
		      (i, test_acc))
        test_acc = sess.run(accuracy, feed_dict = test_feed)
	print("After %d training step, test accuracy is %g" % (TRAINING_STEPS, test_acc))
        test_acc = sess.run(accuracy_avg, feed_dict = test_feed)
	print("After %d training step, test accuracy is %g" % (TRAINING_STEPS, test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets(data_dir, one_hot = True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()
