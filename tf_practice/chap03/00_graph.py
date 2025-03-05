import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    v=tf.get_variable("v1", initializer=tf.zeros_initializer()(shape=[1]))

g2=tf.Graph()
with g2.as_default():
    v=tf.get_variable("v2", initializer=tf.ones_initializer()(shape=[1]))

config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
with tf.Session(graph=g1, config=config) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v1")))

with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v2")))

