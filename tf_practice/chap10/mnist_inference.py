import tensorflow as tf

BATCH_SIZE = 100 
IMAGE_SIZE = 28
NUM_CHANNELS = 1 
OUTPUT_NODE = 10

CONV1_DEEP = 32
CONV1_SIZE = 5
CONV2_DEEP = 64
CONV2_SIZE = 5
FC_SIZE = 512

LEARNING_RATE_BASE = 0.001 
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 2000
MOVING_AVERAGE_DECAY = 0.99

DATA_PATH = "/home/mxd/software/data/MNIST"
MODEL_SAVE_PATH = "/home/mxd/software/github/tf_practice/chap10/models"
MODEL_NAME = "mnist"

def inference(input_tensor, train, regularizer):
    with tf.variable_scope("layer1-conv1"):
        conv1_weights = tf.get_variable(shape = [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], 
	          name = "weight", initializer = tf.truncated_normal_initializer(stddev = 0.05))
	conv1_biases = tf.get_variable(name ="bias", shape = [CONV1_DEEP], initializer =
	                         tf.constant_initializer(0.0))
	conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides = [1, 1, 1, 1], padding = "SAME")
	relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
	        padding = "SAME")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(shape = [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], 
	          name = "weight", initializer = tf.truncated_normal_initializer(stddev = 0.05))
	conv2_biases = tf.get_variable(name = "bias", shape = [CONV2_DEEP], initializer = 
	                         tf.constant_initializer(0.0))
	conv2 = tf.nn.conv2d(pool1, conv2_weights, strides = [1, 1, 1, 1], padding = "SAME")
	relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
	        padding = "SAME")

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope("layer5-fc1"):
        fcl_weights = tf.get_variable(name = "weight", shape = [nodes, FC_SIZE], 
	              initializer = tf.truncated_normal_initializer(stddev = 0.05))
	if regularizer != None:
	    tf.add_to_collection("losses", regularizer(fcl_weights))
	fcl_biases = tf.get_variable(name = "bias", shape = [FC_SIZE], initializer = 
	             tf.constant_initializer(0.1))
	fc1 = tf.nn.relu(tf.matmul(reshaped, fcl_weights) + fcl_biases)
	
	if train:
	    fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope("layer6-fc2"):
        fc2_weights = tf.get_variable(name = "weight", shape = [FC_SIZE, OUTPUT_NODE], 
	              initializer = tf.truncated_normal_initializer(stddev = 0.05))
	fc2_biases = tf.get_variable(name = "bias", shape = [OUTPUT_NODE], initializer = 
	             tf.constant_initializer(0.1))
	if regularizer != None:
	    tf.add_to_collection("losses", regularizer(fc2_weights))
	logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit

