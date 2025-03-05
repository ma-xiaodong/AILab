import tensorflow as tf
import constant_def

v1 = tf.Variable(0, dtype = tf.float32, name = "v1")
v2 = tf.Variable(0, dtype = tf.float32, name = "v2")

add_rlt = v1 + v2
mul_rlt = add_rlt * v2

variable_average = tf.train.ExponentialMovingAverage(constant_def.MOVING_AVERAGE_DECAY)
variable_average_op = variable_average.apply(tf.trainable_variables())
print tf.trainable_variables()

saver = tf.train.Saver()

# write the pb file
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    sess.run(tf.assign(v1, 12))
    sess.run(tf.assign(v2, 32))
    sess.run(variable_average_op)
    print(sess.run([v1, variable_average.average(v1)]))
    print(sess.run([v2, variable_average.average(v2)]))

    sess.run(tf.assign(v1, 34))
    sess.run(tf.assign(v2, 56))
    sess.run(variable_average_op)
    print(sess.run([v1, variable_average.average(v1)]))
    print(sess.run([v2, variable_average.average(v2)]))
    print sess.run(add_rlt)

    saver.save(sess, constant_def.FILE_NAME)

