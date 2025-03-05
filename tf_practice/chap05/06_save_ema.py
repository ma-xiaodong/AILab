import tensorflow as tf

v1 = tf.Variable(0, dtype = tf.float32, name = "v1")

print("global variables:")
for var in tf.global_variables():
    print var

ema = tf.train.ExponentialMovingAverage(0.99)
ema_op = ema.apply([v1])

saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    sess.run(tf.assign(v1, 10))
    sess.run(ema_op)
    saver.save(sess, "/home/mxd/software/github/tensorflow_practice/chap05/models/ema")
    print sess.run([v1, ema.average(v1)])

print ema.variables_to_restore()

saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    sess.run(tf.assign(v1, 24))
    print sess.run(v1)
    saver.restore(sess, "/home/mxd/software/github/tensorflow_practice/chap05/models/ema")
    print sess.run(v1)

