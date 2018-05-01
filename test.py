import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from config import *
import model
np.set_printoptions(suppress=True)

def eval():
    with tf.Graph().as_default():
        fake_data = tf.random_uniform(shape=(FLAGS.batch_size, 100), dtype=tf.float32)
        w_init = tf.contrib.layers.xavier_initializer()
        logits = model.generator(fake_data, False, w_init=w_init) 
        #d_logits = model.discreminator(x_placeholder)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
            saver.restore(sess, ckpt.model_checkpoint_path) 
            image = sess.run(logits)     
            image = image * 0.5 + 0.5
            plt.figure(figsize=(9,1))    
            for k in range(1, 8, 1):
                imgIdx = k
                graphIdx = ''.join(str(x) for x in [1,8,k])
                plt.subplot(graphIdx)
                plt.axis('off')
                plt.imshow(np.reshape(image[imgIdx], [28,28]), cmap="gray")
            
            plt.show()


#data_sets = read_data.read_data()
#images, _ = data_sets.train.next_batch(FLAGS.batch_size, FLAGS.fake_data)
#test_data = np.random.uniform(-1, 1, size=(1, 100))
eval()
