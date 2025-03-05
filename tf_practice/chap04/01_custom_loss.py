import tensorflow as tf
from numpy.random import RandomState

batch_size = 8
x = tf.placeholder(dtype=tf.float32, shape=(None, 2), name = "x-input")
y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1), name = "y-input")

w1 = tf.Variable(tf.random_normal([2, 1], stddev = 1, seed = 1))
y = tf.matmul(x, w1)
loss_less = 1
loss_more = 10

loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

rdm = RandomState(1)
data_size = 128
X = rdm.rand(data_size, 2)
Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    STEPS = 10000
    print("Initial params:")
    print w1.eval()
    for i in range(STEPS):
	start = (i * batch_size) % data_size
	end = min(start + batch_size, data_size)
	sess.run(train_step, feed_dict = {x: X[start: end], y_: Y[start: end]})

	if(i % 100 == 0):
	    total_loss = sess.run(loss, feed_dict = {x: X, y_: Y})
	    print("Traning No. %d, total loss: %g" % (i, total_loss))
    print("Final params: ")
    print w1.eval()

