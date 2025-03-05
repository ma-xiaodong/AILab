import tensorflow as tf

filenames = tf.train.match_filenames_once('00*-of-00*')
filenames_queue = tf.train.string_input_producer(filenames, shuffle = False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filenames_queue)
features = tf.parse_single_example(serialized_example, 
    features = {
    'i': tf.FixedLenFeature([], tf.int64),
    'j': tf.FixedLenFeature([], tf.int64),
    })

with tf.Session() as sess:
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
    print sess.run(filenames)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    for i in range(6):
	print sess.run([features['i'], features['j']])
    coord.request_stop()
    coord.join(threads)

