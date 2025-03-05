import tensorflow as tf

que = tf.FIFOQueue(3, 'int32')

init = que.enqueue_many(([1, 4, 9], ))
x = que.dequeue()

y = x + 1
q_inc = que.enqueue([y])

with tf.Session() as sess:
    init.run()
    for _ in range(5):
        v, _ = sess.run([x, q_inc])
        print v

