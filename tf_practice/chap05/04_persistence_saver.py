import tensorflow as tf

v1 = tf.Variable(tf.constant(135.0, shape = [1]), name = "v1")
v2 = tf.Variable(tf.constant(246.0, shape = [1]), name = "v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init_op.run()
    saver.save(sess, "/home/mxd/software/github/tensorflow_practice/chap05/models/mnist")
