import tensorflow as tf

def assign_device():
    with tf.device('/cpu:0'):
        a = tf.constant([1, 2, 3], shape = [3], dtype = tf.float32, name = 'a')
        b = tf.constant([4, 5, 6], shape = [3], dtype = tf.float32, name = 'b')

    with tf.device('/gpu:0'):
        c = a + b

    with tf.Session(config = tf.ConfigProto(log_device_placement = True)) as sess:
        print sess.run(c)

def soft_placement():
    a_cpu = tf.Variable(0, name = 'a_cpu')

    with tf.device('/gpu:0'):
        b_gpu = tf.Variable(0, name = 'b_gpu')

    with tf.Session(config = tf.ConfigProto(allow_soft_placement = False, 
        log_device_placement = False)) as sess:
        tf.global_variables_initializer().run()
	print sess.run(a_cpu)
	print sess.run(b_gpu)

def main(argv = None):
    assign_device()
    soft_placement()

if __name__ == '__main__':
    main()
