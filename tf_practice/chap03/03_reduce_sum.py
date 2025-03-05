import tensorflow as tf

import numpy as np 

x = np.asarray([[[1,1,1,1],[1,1,1,1],[1,1,1,1]], [[1,1,1,1],[1,1,1,1],[1,1,1,1]]])
x_p = tf.placeholder(dtype = tf.int32, shape = (2, 3, 4), name = "x_p")

y = tf.reduce_sum(x_p, reduction_indices = 1)
with tf.Session() as sess:
    print(sess.run(y, feed_dict = {x_p : x}))
 
