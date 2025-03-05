from tensorflow.examples.tutorials.mnist import input_data
data_dir = "/home/mxd/software/data/MNIST"

mnist = input_data.read_data_sets(data_dir, one_hot = True)
print "Training data size: ", mnist.train.num_examples
print "Validation data size: ", mnist.validation.num_examples
print "Testing data size: ", mnist.test.num_examples

print "Example of training data: "
print mnist.train.images[0]
print "Example of training label: "
print mnist.train.labels[0]

batch_size = 128
xs, ys = mnist.train.next_batch(batch_size)
print xs.shape
print ys.shape

