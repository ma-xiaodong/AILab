import tensorflow as tf
from tensorflow.python.platform import gfile

FILE_NAME = "/home/mxd/software/github/tensorflow_practice/chap05/add.pb"

with tf.Session() as sess:
    with gfile.FastGFile(FILE_NAME, "rb") as f:
        graph = tf.GraphDef()
	graph.ParseFromString(f.read())
    result = tf.import_graph_def(graph, return_elements = ["add:0"])
    print sess.run(result)
