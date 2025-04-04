import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(name = "weights", shape = shape, initializer = 
                              tf.truncated_normal_initializer(stddev = 0.05))
    if regularizer != None:
	tf.add_to_collection("losses", regularizer(weights))
    return weights

def inference(input_tensor, regularizer):
    with tf.variable_scope("layer1"):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
	biases = tf.get_variable(name ="biases", shape = [LAYER1_NODE], initializer =
	                         tf.constant_initializer(0.0))
	layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope("layer2"):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
	biases = tf.get_variable(name = "biases", shape = [OUTPUT_NODE], initializer = 
	                         tf.constant_initializer(0.0))
	layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
    return layer2




