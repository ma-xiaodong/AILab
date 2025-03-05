import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

def train(mnist):
    x = tf.placeholder(shape = [mnist_inference.BATCH_SIZE, mnist_inference.IMAGE_SIZE,
        mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS], dtype = tf.float32, name = "x-input")
    y_ = tf.placeholder(shape = [mnist_inference.BATCH_SIZE, mnist_inference.OUTPUT_NODE], 
         dtype = tf.float32, name = "y-input")

    regularizer = tf.contrib.layers.l2_regularizer(mnist_inference.REGULARIZATION_RATE)
    y = mnist_inference.inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable = False)

    variable_averages = tf.train.ExponentialMovingAverage(mnist_inference.MOVING_AVERAGE_DECAY)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    learning_rate = tf.train.exponential_decay(mnist_inference.LEARNING_RATE_BASE, global_step,
		    mnist.train.num_examples / mnist_inference.BATCH_SIZE, mnist_inference.LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name = "train")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
	for i in range(mnist_inference.TRAINING_STEPS):
	    xs, ys = mnist.train.next_batch(mnist_inference.BATCH_SIZE)
	    reshape_xs = np.reshape(xs, (mnist_inference.BATCH_SIZE, mnist_inference.IMAGE_SIZE, 
	                 mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
	    _, loss_value, step = sess.run([train_op, loss, global_step],
	                                   feed_dict = {x: reshape_xs, y_: ys})
	    if i % 1000 == 0:
		print "After %d training, loss on training batch is %g" % (step, loss_value)
		saver.save(sess, os.path.join(mnist_inference.MODEL_SAVE_PATH, mnist_inference.MODEL_NAME),
		           global_step = global_step)

def main():
    mnist = input_data.read_data_sets(mnist_inference.DATA_PATH, one_hot = True)
    train(mnist)

if __name__ == "__main__":
    main()

