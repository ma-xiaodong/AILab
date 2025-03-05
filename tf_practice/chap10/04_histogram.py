import tensorflow as tf

input_size = 100

X = tf.placeholder(tf.float32, [None, input_size], name="input_x")
x_image = tf.reshape(X, [-1, 6, 10, 1])
tf.summary.image('input', x_image, 4)

# First layer of weights
with tf.name_scope("layer1"):
    W1 = tf.get_variable("W1", shape=[input_size, hidden_layer_neurons],
                         initializer=tf.contrib.layers.xavier_initializer())
    layer1 = tf.matmul(X, W1)
    layer1_act = tf.nn.tanh(layer1)
    tf.summary.histogram("weights", W1)
    tf.summary.histogram("layer", layer1)
    tf.summary.histogram("activations", layer1_act)
# Second layer of weights
with tf.name_scope("layer2"):
    W2 = tf.get_variable("W2", shape=[hidden_layer_neurons, hidden_layer_neurons],
                         initializer=tf.contrib.layers.xavier_initializer())
    layer2 = tf.matmul(layer1_act, W2)
    layer2_act = tf.nn.tanh(layer2)
    tf.summary.histogram("weights", W2)
    tf.summary.histogram("layer", layer2)
    tf.summary.histogram("activations", layer2_act)

# Third layer of weights
with tf.name_scope("layer3"):
    W3 = tf.get_variable("W3", shape=[hidden_layer_neurons, hidden_layer_neurons],
                         initializer=tf.contrib.layers.xavier_initializer())
    layer3 = tf.matmul(layer2_act, W3)
    layer3_act = tf.nn.tanh(layer3)
    tf.summary.histogram("weights", W3)
    tf.summary.histogram("layer", layer3)
    tf.summary.histogram("activations", layer3_act)

# Fourth layer of weights
with tf.name_scope("layer4"):
    W4 = tf.get_variable("W4", shape=[hidden_layer_neurons, output_size],
                         initializer=tf.contrib.layers.xavier_initializer())
    Qpred = tf.nn.softmax(tf.matmul(layer3_act, W4)) # Bug fixed: Qpred = tf.nn.softmax(tf.matmul(layer3, W4))
    tf.summary.histogram("weights", W4)
    tf.summary.histogram("Qpred", Qpred)

# We need to define the parts of the network needed for learning a policy
Y = tf.placeholder(tf.float32, [None, output_size], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

# Loss function
# Sum (Ai*logp(yi|xi))
log_lik = -Y * tf.log(Qpred)
loss = tf.reduce_mean(tf.reduce_sum(log_lik * advantages, axis=1))
tf.summary.scalar("Q", tf.reduce_mean(Qpred))
tf.summary.scalar("Y", tf.reduce_mean(Y))
tf.summary.scalar("log_likelihood", tf.reduce_mean(log_lik))
tf.summary.scalar("loss", loss)

# Learning
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


