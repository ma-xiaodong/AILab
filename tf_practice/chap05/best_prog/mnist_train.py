import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 40000
MOVING_AVERAGE_DECAY = 0.99
DATA_PATH = "/home/mxd/software/data/MNIST"
MODEL_SAVE_PATH = "/home/mxd/software/github/tensorflow_practice/chap05/models"
MODEL_NAME = "mnist_best"

def train(mnist):
    x = tf.placeholder(shape = [None, mnist_inference.INPUT_NODE], 
                       dtype = tf.float32, name = "x-input")
    y_ = tf.placeholder(shape = [None, mnist_inference.OUTPUT_NODE],
                        dtype = tf.float32, name = "y-input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable = False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, )
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, 
                    labels = tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    learning_rate = tf.train.exponential_decay(
                    LEARNING_RATE_BASE, global_step,
		    mnist.train.num_examples / BATCH_SIZE,
		    LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
                 .minimize(loss, global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name = "train")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
	for i in range(TRAINING_STEPS):
	    xs, ys = mnist.train.next_batch(BATCH_SIZE)
	    _, loss_value, step = sess.run([train_op, loss, global_step],
	                                   feed_dict = {x: xs, y_: ys})
	    if i % 1000 == 0:
		print "After %d training, loss on training batch is %g" % (step, loss_value)
		saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
		           global_step = global_step)

def main():
    mnist = input_data.read_data_sets(DATA_PATH, one_hot = True)
    train(mnist)

if __name__ == "__main__":
    main()

