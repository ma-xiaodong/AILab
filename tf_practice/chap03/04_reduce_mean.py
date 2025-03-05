import numpy as np 
import tensorflow as tf 


x = np.array([[1.,2.,3.],[4.,5.,6.]]) 

with tf.Session() as sess:
    mean_none = sess.run(tf.reduce_mean(x)) 
    mean_0 = sess.run(tf.reduce_mean(x, 0)) 
    mean_1 = sess.run(tf.reduce_mean(x, 1)) 

    print (x) 
    print (mean_none)
    print (mean_0) 
    print (mean_1) 
