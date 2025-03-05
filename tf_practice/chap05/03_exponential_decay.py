import tensorflow as tf
import numpy as np

learning_rate = 0.1
decay_rate = 0.96
global_steps = 100
decay_steps = 20

global_ = tf.Variable(tf.constant(0))
c = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase = True)
d = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase = False)
T_C = []
F_D = []

with tf.Session() as sess, tf.device('/cpu:0'):
    for i in range(global_steps):
	T_c = sess.run(c, feed_dict = {global_: i})
	T_C.append(T_c)
	F_d = sess.run(d, feed_dict = {global_: i})
	F_D.append(F_d)
print("values of T_C: ")
i = 0
for x in T_C:
    print x,
    i = i + 1
    if i % 10 == 0:
	print
print("values of F_D: ")
i = 0
for x in F_D:
    print x,
    i = i + 1
    if i % 10 == 0:
	print
