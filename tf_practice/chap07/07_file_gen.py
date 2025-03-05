import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def main(argv = None):
    num_files = 2
    num_instance = 2

    for i in range(num_files):
	writer = tf.python_io.TFRecordWriter('./%.5d-of-%.5d' % (i, num_files))
	for j in range(num_instance):
	    example = tf.train.Example(features = tf.train.Features(
	        feature = {'i': _int64_feature(i), 'j': _int64_feature(j)}))
	    writer.write(example.SerializeToString())
	writer.close()

if __name__ == '__main__':
    tf.app.run()
