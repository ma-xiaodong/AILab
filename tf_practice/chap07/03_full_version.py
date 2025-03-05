import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pdb

def preprocess_for_train(image, height, width, bbox):
    pdb.set_trace()
    if bbox is None:
	bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype = tf.float32, shape = [1, 1, 4])
    if image.dtype != tf.float32:
	image = tf.image.convert_image_dtype(image, dtype = tf.float32)

    image = tf.image.resize_images(image, [height, width], method = 1)
    bbox_begin, bbox_size, bboxes = tf.image.sample_distorted_bounding_box(
        [image.eval().shape[0], image.eval().shape[1], image.eval().shape[2]],
	bounding_boxes = bbox)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

def main(argv = None):
    img_raw_data = tf.gfile.FastGFile('./mxd.jpg', 'r').read()
    with tf.Session() as sess:
        image_data = tf.image.decode_jpeg(img_raw_data)
        boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7],]])

	for i in range(6):
	    image = preprocess_for_train(image_data, 200, 200, boxes)
	    plt.imshow(image.eval())
	    plt.show()    

if __name__ == '__main__':
    tf.app.run()
