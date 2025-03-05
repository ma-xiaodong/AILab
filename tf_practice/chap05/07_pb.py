import tensorflow as tf
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(12.0, shape = [1]), name = "v1")
v2 = tf.Variable(tf.constant(34.0, shape = [1]), name = "v2")
add_rlt = v1 + v2
mul_rlt = add_rlt * v2
print add_rlt.name, mul_rlt.name

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    init_op.run()
    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(sess,
                       graph_def, ["add"])
    with tf.gfile.GFile("/home/mxd/software/github/tensorflow_practice/chap05/add.pb", 
                        "wb") as f:
        f.write(output_graph_def.SerializeToString())
