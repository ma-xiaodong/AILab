import tensorflow as tf
import pdb

labels = tf.Variable(name = "label", initial_value = tf.truncated_normal([5, 3], stddev = 5.1)) 
logits = tf.Variable(name = "logits", initial_value = tf.truncated_normal([5, 3], stddev = 4.9)) 
pdb.set_trace()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print "labels:"
    print labels.eval()
    print "logits:"
    print logits.eval()
    print "argmax(logits):"
    print tf.argmax(logits, 1).eval()
    print "equals:"
    equals = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
    print equals.eval()
    print "equals reduce mean:"
    print tf.reduce_mean(tf.cast(equals, tf.float32)).eval()
    print "cross_entropy_mean:"
    print tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels).eval()
    print "sparse_cross_entropy_mean:"
    print tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = tf.argmax(labels, 1)).eval()

