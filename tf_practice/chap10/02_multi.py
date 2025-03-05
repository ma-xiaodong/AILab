import os
import time
from datetime import datetime

import tensorflow as tf
import mnist_inference
import pdb

FILE_NAME = '/home/mxd/software/github/tf_practice/chap10/output.tfrecords'
N_GPU = 1

def get_input():
    filename_queue = tf.train.string_input_producer([FILE_NAME])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features = {
	'image_raw': tf.FixedLenFeature([], tf.string),
	'pixels': tf.FixedLenFeature([], tf.int64),
	'labels': tf.FixedLenFeature([], tf.int64)})

    decoded_image = tf.decode_raw(features['image_raw'], tf.uint8)
    reshaped_image = tf.reshape(decoded_image, [28, 28, 1])
    retyped_image = tf.cast(reshaped_image, tf.float32)
    label = tf.cast(features['labels'], tf.int32)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * mnist_inference.BATCH_SIZE

    return tf.train.shuffle_batch(
        [retyped_image, label],
	batch_size = mnist_inference.BATCH_SIZE,
	capacity = capacity,
	min_after_dequeue = min_after_dequeue)

def get_loss(x, y_, regularizer, scope, reuse_variables = None):
    with tf.variable_scope(tf.get_variable_scope(), reuse = reuse_variables):
        y = mnist_inference.inference(x, True, regularizer)
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = y_))
    regularization_loss = tf.add_n(tf.get_collection('losses', scope))
    loss = regularization_loss + cross_entropy
    return loss

def average_gradients(tower_grads):
    average_grads = []

    for grad_and_vars in zip(*tower_grads):
	grads = []
	for g, _ in grad_and_vars:
	    expanded_g = tf.expand_dims(g, 0)
	    grads.append(expanded_g)
	grad = tf.concat(grads, 0)
	grad = tf.reduce_mean(grad, 0)

	v = grad_and_vars[0][1]
	grad_and_var = (grad, v)
	average_grads.append(grad_and_var)
    return average_grads

def main(argv = None):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        x, y_ = get_input()
        regularizer = tf.contrib.layers.l2_regularizer(
	    mnist_inference.REGULARIZATION_RATE)
	global_step = tf.get_variable(name = 'global_step', shape = [], 
	    initializer = tf.constant_initializer(0), trainable = False)
	learning_rate = tf.train.exponential_decay(
	    mnist_inference.LEARNING_RATE_BASE, global_step, 
	    60000 / mnist_inference.BATCH_SIZE, 
	    mnist_inference.LEARNING_RATE_DECAY)

	opt = tf.train.GradientDescentOptimizer(learning_rate)
	tower_grads = []
	reuse_variables = False

        for i in range(N_GPU):
	    with tf.device('/gpu:%d' %i):
	        with tf.name_scope('GPU_%d' %i) as scope:
	            cur_loss = get_loss(x, y_, regularizer, scope, reuse_variables)
		    reuse_variables = True
		    grads = opt.compute_gradients(cur_loss)
		    tower_grads.append(grads)
	grads = average_gradients(tower_grads)
        # because I have only one gpu, no use to call average_gradients 
	#grads = tower_grads[0]

	for grad, var in grads:
	    if grad is not None:
		tf.summary.histogram('grad_on_avg/%s' % var.op.name, grad)

        apply_gradients_op = opt.apply_gradients(grads, global_step = global_step)
	for var in tf.trainable_variables():
	    tf.summary.histogram(var.op.name, var)
	
	variable_average = tf.train.ExponentialMovingAverage(mnist_inference.MOVING_AVERAGE_DECAY, global_step)
	variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
	variable_average_op = variable_average.apply(variables_to_average)

	train_op = tf.group(apply_gradients_op, variable_average_op)

	saver = tf.train.Saver(tf.all_variables())
	summary_op = tf.summary.merge_all()

	with tf.Session(config = tf.ConfigProto(allow_soft_placement = True,
	    log_device_placement = False)) as sess:
            tf.global_variables_initializer().run()

	    coord = tf.train.Coordinator()
	    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
	    summary_writer = tf.summary.FileWriter(mnist_inference.MODEL_SAVE_PATH, sess.graph)

	    for step in range(mnist_inference.TRAINING_STEPS):
		start_time = time.time()
		_, loss_value = sess.run([train_op, cur_loss])
		duration = time.time() - start_time

		if step != 0 and step % 10 == 0:
		    num_examples_per_step = mnist_inference.BATCH_SIZE * N_GPU
		    examples_per_sec = num_examples_per_step / duration
		    sec_per_batch = duration / N_GPU

		    format_str = ('step %d, loss = %.2f (%.1f examples / '
		        ' sec; %.3f sec / batch)')
		    print(format_str % (step, loss_value, examples_per_sec,
		        sec_per_batch))

		    summary = sess.run(summary_op)
		    summary_writer.add_summary(summary, step)
		if step % 1000 == 0 or (step + 1) == mnist_inference.TRAINING_STEPS:
		    checkpoint_path =  os.path.join(mnist_inference.MODEL_SAVE_PATH, 
			mnist_inference.MODEL_NAME)
		    saver.save(sess, checkpoint_path, global_step = step)
            coord.request_stop()
	    coord.join(threads)

if __name__ == '__main__':
    tf.app.run()

