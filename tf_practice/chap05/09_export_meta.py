import tensorflow as tf
FILE_NAME = "/home/mxd/software/github/tensorflow_practice/chap05/add.json"

v1 = tf.Variable(tf.constant(1.2, shape = [1]), dtype = tf.float32, name = "v1")
v2 = tf.Variable(tf.constant(2.3, shape = [1]), dtype = tf.float32, name = "v2")
result = v1 + v2

saver = tf.train.Saver()
saver.export_meta_graph(FILE_NAME, as_text = True)
