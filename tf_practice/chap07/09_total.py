import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math
import pdb

DATA_PATH = "/home/mxd/software/data/MNIST"
FILE_NAME = './output.tfrecords'

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def write_tfrecord():
    mnist = input_data.read_data_sets(DATA_PATH, dtype = tf.uint8, one_hot = True)
    images = mnist.train.images
    labels = mnist.train.labels
    pixels = images.shape[1]
    num_examples = mnist.train.num_examples

    writer = tf.python_io.TFRecordWriter(FILE_NAME)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features = tf.train.Features(feature = {
            'pixels': _int64_feature(pixels),
	    'labels': _int64_feature(np.argmax(labels[index])),
	    'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

def read_tfrecord():
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([FILE_NAME])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, 
        features = {
	    'image_raw': tf.FixedLenFeature([], tf.string),
	    'pixels': tf.FixedLenFeature([], tf.int64),
	    'labels': tf.FixedLenFeature([], tf.int64)
	})
    decoded_image = tf.decode_raw(features['image_raw'], tf.uint8)
    pdb.set_trace()
    label = tf.cast(features['labels'], tf.int32)
    pixel = tf.cast(features['pixels'], tf.float32)
    
    height = tf.sqrt(pixel)
    width = height
    # channel is equal 1
    decoded_image.set_shape([height, width, 1])

def main(argv = None):
    write_tfrecord()
    # read_tfrecord()

if __name__ == '__main__':
    tf.app.run()

