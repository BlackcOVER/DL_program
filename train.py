import tensorflow as tf
import model
import read_data 
import argparse 
from config import *
import time
import os
import pandas as pd
import numpy as np
from utils import *
import matplotlib.pyplot as plt

def get_placeholder(dim):
    x_placeholder = tf.placeholder(tf.float32, shape=(None, dim))
    return x_placeholder

def train():
    truth_placeholder = get_placeholder(784)
    #fake_data = tf.random_uniform(shape=(FLAGS.batch_size, 100), dtype=tf.float32)
    fake_placeholder = get_placeholder(100)
    w_init = tf.contrib.layers.xavier_initializer() 
    #g_logits = model.generator(fake_data, False, w_init)
    g_logits = model.generator(fake_placeholder, False, w_init)

    y_truth, d_t_logits = model.discreminator(truth_placeholder, False, w_init)
    y_fake, d_f_logits = model.discreminator(g_logits, True, w_init)

    g_loss = model.g_loss(d_f_logits)
    d_loss = model.d_loss(d_t_logits, d_f_logits)

    train_g_op, train_d_op = model.train(g_loss, d_loss, FLAGS.d_learning_rate, FLAGS.g_learning_rate) 
    
    data_sets = read_data.read_data()    
    images_size = data_sets.train.images.shape[0]
    steps_per_epoch = images_size // FLAGS.batch_size

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init)
        for step in xrange(FLAGS.max_epochs * steps_per_epoch):
            fake_data = tf.random_uniform(shape=(FLAGS.batch_size, 100), dtype=tf.float32)
            fake_data = sess.run(fake_data)
            start_time = time.time() 
            truth_data, label = data_sets.train.next_batch(FLAGS.batch_size, FLAGS.fake_data)
            truth_data = (truth_data - 0.5) / 0.5 
            #_, mid_g_loss = sess.run([train_g_op, g_loss])
            _, mid_g_loss = sess.run([train_g_op, g_loss], feed_dict={fake_placeholder: fake_data})
            for i in xrange(5):
                _, mid_d_loss = sess.run([train_d_op, d_loss], feed_dict={truth_placeholder: truth_data, fake_placeholder: fake_data})
            duration = time.time() - start_time
            if step % 100 == 0:
                print 'Step: %d, d_loss: %f, g_loss:%f, time: %f' % (step, mid_d_loss, mid_g_loss, duration)    
            if step % 200 == 0:
                fake_data_test = tf.random_uniform(shape=(FLAGS.batch_size, 100), dtype=tf.float32)
                g_logits = model.generator(fake_data_test, True, w_init)
                images = sess.run(g_logits)
                images = images / 2 + 0.5
                plt.figure(figsize=(9,1))
                for k in range(1, 8, 1):
                    imgIdx = k
                    graphIdx = ''.join(str(x) for x in [1,8,k])
                    plt.subplot(graphIdx)
                    plt.axis('off')
                    plt.imshow(np.reshape(images[imgIdx], [28,28]), cmap="gray")
                plt.show()
            if (step+1) % 1000 == 0 or (step+1) == (FLAGS.max_epochs * steps_per_epoch):
                checkpoint_file = os.path.join(FLAGS.ckpt_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
     
    
if __name__ == '__main__':
    train()
