import tensorflow as tf
from numpy.random import RandomState

w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))

x = tf.placeholder(dtype = tf.float32, shape = (None, 2), name = "input")
y_ = tf.placeholder(dtype = tf.float32, shape = (None, 1), name = "output")

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

rdm = RandomState(1)
batch_sz = 10
dataset_sz = 256
X = rdm.rand(dataset_sz, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print sess.run(w1)
    print sess.run(w2)
    steps = 1000
    for i in range(steps):
	start = (i * batch_sz) % dataset_sz
	end = min(start + batch_sz, dataset_sz)
	sess.run(train_step, feed_dict = {x: X[start: end], y_: Y[start: end]})

	if i % 100 == 0:
	    total_cross_entropy = sess.run(cross_entropy, feed_dict = 
	                          {x: X, y_: Y})
	    print("After %d training steps, cross entropy on all data is %g"
	           % (i, total_cross_entropy))
    print sess.run(w1)
    print sess.run(w2)

