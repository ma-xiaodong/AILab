import tensorflow as tf
import constant_def
import pdb

v1 = tf.Variable(0, dtype = tf.float32, name = "v1")
v2 = tf.Variable(0, dtype = tf.float32, name = "v2")
result = v1 + v2

variables_average = tf.train.ExponentialMovingAverage(constant_def.MOVING_AVERAGE_DECAY)
variables_to_restore = variables_average.variables_to_restore()
print variables_to_restore

saver = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(constant_def.FILE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
	saver.restore(sess, constant_def.FILE_NAME)
	print sess.run(result)
    else:
	print "Not found ckpt!"
    
