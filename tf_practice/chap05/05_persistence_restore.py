import tensorflow as tf

saver = tf.train.import_meta_graph(
        "/home/mxd/software/github/tensorflow_practice/chap05/models/mnist.meta")
with tf.Session() as sess:
    saver.restore(sess, "/home/mxd/software/github/tensorflow_practice/chap05/models/")
    print sess.run(tf.get_default_graph().get_tensor_by_name("add:0"))
