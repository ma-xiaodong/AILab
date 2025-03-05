import tensorflow as tf 

N = 20 
k = tf.placeholder(tf.float32)

mean_moving_0 = tf.random_normal(shape=[500], mean=(5*k), stddev=1) 
tf.summary.histogram("normal/moving_mean_0", mean_moving_0) 

mean_moving_1 = tf.random_normal(shape=[1000], mean=(5*k), stddev=5) 
tf.summary.histogram("normal/moving_mean_1", mean_moving_1) 

summaries = tf.summary.merge_all() 

sess = tf.Session() 
writer = tf.summary.FileWriter("./logs") 

for step in range(N): 
    k_val = step/float(N) 
    summ = sess.run(summaries, feed_dict={k: k_val}) 
    writer.add_summary(summ, global_step=step)

sess.close()
