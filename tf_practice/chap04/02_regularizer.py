import tensorflow as tf
from numpy.random import RandomState

def get_weight(shape, lamb):
    var = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
    tf.add_to_collection("loss", tf.contrib.layers.l2_regularizer(lamb)(var))
    return var

x = tf.placeholder(dtype = tf.float32, shape = (None, 2))
y_ = tf.placeholder(dtype = tf.float32, shape = (None, 1))

batch_size = 8
layer_dimension = [2, 10, 10, 10, 1]
n_layers = len(layer_dimension)

current_layer = x
in_dimension = layer_dimension[0]

for i in range(1, n_layers):
    out_dimension = layer_dimension[i]
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape = [out_dimension]), dtype = tf.float32)
    current_layer = tf.nn.relu(tf.matmul(current_layer, weight) + bias)
    in_dimension = layer_dimension[i]

mse_loss = tf.reduce_mean(tf.square(current_layer - y_))
tf.add_to_collection("loss", mse_loss)
loss = tf.add_n(tf.get_collection("loss"))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# generating the input data
dataset_size = 1200
rdm = RandomState(1)
X = rdm.rand(dataset_size, 2)
Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05]  for (x1, x2) in X]

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    STEPS = 4000
    for i in range(STEPS):
	start = (i * batch_size) % dataset_size
	end = min(start + batch_size, dataset_size)
	sess.run(train_step, feed_dict = {x: X[start: end], y_: Y[start: end]})

	if i % 100 == 0:
	    print "Total mse loss: ", sess.run(mse_loss, feed_dict = {x: X, y_: Y})

