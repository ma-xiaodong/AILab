import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))

x = tf.placeholder(dtype = tf.float32, shape = (3, 2), name = "input")

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# way 1
with tf.Session() as sess:
  sess.run(w1.initializer)
  sess.run(w2.initializer)
  print sess.run(y, feed_dict = {x: [[1, 2], [3, 4], [5, 6]]})

# way 2
with tf.Session() as sess:
  tf.global_variables_initializer().run()
  print y.eval(feed_dict = {x: [[1, 2], [3, 4], [5, 6]]})
