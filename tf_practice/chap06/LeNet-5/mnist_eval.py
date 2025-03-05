import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import numpy as np

import mnist_inference
TIME_INTERVAL = 10
VALIDATE_SIZE = 5000

def evaluate(mnist):
    x = tf.placeholder(shape = [VALIDATE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE,
        mnist_inference.NUM_CHANNELS], dtype = tf.float32, name = "x-input")
    y_ = tf.placeholder(shape = [VALIDATE_SIZE, mnist_inference.OUTPUT_NODE], 
         dtype = tf.float32, name = "y-input")
    xv = mnist.validation.images
    reshape_xv = np.reshape(xv, [VALIDATE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, 1])

    y = mnist_inference.inference(x, False, None)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    variable_averages = tf.train.ExponentialMovingAverage(mnist_inference.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)
    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_inference.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                accuracy_store = sess.run(accuracy, feed_dict = {x: reshape_xv, y_: mnist.validation.labels})
                print("After training %s steps, validation accuracy = %g" % (global_step, accuracy_store))
            else:
                print "No checkpoint file found!"
                return
	time.sleep(TIME_INTERVAL)

def main(argv = None):
    mnist = input_data.read_data_sets(mnist_inference.DATA_PATH, one_hot = True)
    evaluate(mnist)

if __name__ == "__main__":
    tf.app.run()
