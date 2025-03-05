import tensorflow as tf
import numpy as np
import threading
import time

def MyLoop(coord, worker_id):
    while not coord.should_stop():
	if np.random.rand() < 0.1:
	    coord.request_stop()
	    print 'Stopping from %d id...' % (worker_id)
	else:
	    print 'Id %d working.' % (worker_id)
	time.sleep(1)

def main(argv = None):
    coord = tf.train.Coordinator()
    threads = [threading.Thread(target = MyLoop, args = (coord, i, ))
        for i in xrange(5)]
    for t in threads:
	t.start()
    coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
