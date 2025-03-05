import tensorflow as tf

def main(argv = None):
    input_tensor = tf.get_variable(shape = [1, 5, 5, 1], name = "image", 
                   initializer = tf.truncated_normal_initializer(stddev = 10))
    filter_weights = tf.get_variable(shape = [3, 3, 1, 1], name = "weights",
                     initializer = tf.truncated_normal_initializer(stddev = 0.1))
    biases = tf.get_variable(shape = [1], name = "biases", initializer = 
                             tf.constant_initializer(0.1))
    conv = tf.nn.conv2d(input_tensor, filter_weights, strides = [1, 1, 1, 1],
           padding = "SAME")
    bias = tf.nn.bias_add(conv, biases)
    pool = tf.nn.max_pool(bias, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
           padding = "SAME")

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
	print "input_tensor: "
	print input_tensor.eval()
	print "filter_weights: "
	print filter_weights.eval()
	print "conv result: "
	print bias.eval()
	print "pooling result"
	print pool.eval()

if __name__ == "__main__":
    tf.app.run()
