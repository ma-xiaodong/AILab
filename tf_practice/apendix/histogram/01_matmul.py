import tensorflow as tf

with tf.name_scope('graph') as scope:
    matrix1 = tf.constant([[3., 3.]], name = 'matrix1')
    matrix2 = tf.constant([[2.], [4.]], name = 'matrix2')
    product = tf.matmul(matrix1, matrix2, name = 'produce')

with tf.Session() as sess:
    tf.global_variables_initializer().run
    sess.run(product)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    
