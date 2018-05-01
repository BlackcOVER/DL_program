import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from config import *

def gen_fake_data():
    fake_data = np.random.uniform(-1, 1, size=(FLAGS.batch_size, 100))
    #fake_data = np.random.randn(FLAGS.batch_size, 100) 
    return fake_data

def read_data(is_fake=None):
    if is_fake:
        fake_data = gen_fake_data()
        return fake_data
    else:
        data_sets = input_data.read_data_sets(FLAGS.input_dir, FLAGS.fake_data)
        images, labels =  data_sets.train.next_batch(FLAGS.batch_size, FLAGS.fake_data)
        return data_sets

 
if __name__ == '__main__':
    #fake_data = read_data(True)
    #fake_data_batch = tf.train.batch([fake_data], batch_size=10, capacity=20)
    #with tf.Session() as sess:
    #    init = tf.global_variables_initializer()
    #    sess.run(init)
    #    print sess.run(fake_data_batch) 
    data_set = read_data()    
