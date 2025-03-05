import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 6

def evaluate(mnist):
    x = tf.placeholder(shape = [None, mnist_inference.INPUT_NODE], dtype = tf.float32, name = "x-input")
    y_ = tf.placeholder(shape = [None, mnist_inference.OUTPUT_NODE], dtype = tf.float32, name = "y-input")
    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

    y = mnist_inference.inference(x, None)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    print "variables_to_restore: ", variables_to_restore

    saver = tf.train.Saver(variables_to_restore)

    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
	    if ckpt and ckpt.model_checkpoint_path:
	        saver.restore(sess, ckpt.model_checkpoint_path)
	        global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
	        accuracy_store = sess.run(accuracy, feed_dict = validate_feed)
	        print("After training %s steps, validation accuracy = %g" % (global_step, accuracy_store))
	    else:
	        print "No checkpoint file found!"
	        return
        time.sleep(EVAL_INTERVAL_SECS)

def main(argv = None):
    mnist = input_data.read_data_sets(mnist_train.DATA_PATH, one_hot = True)
    evaluate(mnist)

if __name__ == "__main__":
    tf.app.run()
