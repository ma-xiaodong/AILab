import matplotlib.pyplot as plt
import tensorflow as tf
import pdb

image_raw_data = tf.gfile.FastGFile('./mxd.jpg').read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    img_data = tf.image.resize_images(img_data, [1280, 960], method = 1)
    #plt.imshow(img_data.eval())
    #plt.show()
    
    img_data_f = tf.image.convert_image_dtype(img_data, tf.float32)
    batched = tf.expand_dims(img_data_f, 0)
    boxes = tf.constant([[[0.05, 0.5, 0.8, 0.9]]])
    result = tf.image.draw_bounding_boxes(batched, boxes)
    plt.imshow(result[0].eval())
    plt.show()

    encoded_image = tf.image.encode_jpeg(img_data)

    with tf.gfile.GFile('./mxd.out.jpg', 'wb') as f:
        f.write(encoded_image.eval())

