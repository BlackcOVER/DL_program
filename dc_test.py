import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from config import *
import dc_model as model
from utils import *
np.set_printoptions(suppress=True)

def get_placeholder(dim):
    if type(dim) == int:
        x_placeholder = tf.placeholder(tf.float32, shape=(None, dim))
    else:
        x_placeholder = tf.placeholder(tf.float32, shape=dim)
    return x_placeholder

def eval():
    with tf.Graph().as_default():
        fake_placeholder = tf.placeholder(tf.float32, shape=(None, 100))
        label_placeholder = get_placeholder(10)
        logits = model.generator(fake_placeholder)
        with tf.Session() as sess:
            fake_data = np.random.uniform(size=(FLAGS.batch_size, 100))
            #label = np.random.randint(10,size=[FLAGS.batch_size, 1])
            #print label[:10]
            #label = one_hot(label)
            #label = sess.run(label)
            #label = label.reshape([FLAGS.batch_size, 10])
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(FLAGS.dc_ckpt_dir)
            saver.restore(sess, ckpt.model_checkpoint_path) 
            image = sess.run(logits, feed_dict={fake_placeholder: fake_data})     
            image = image * 0.5 + 0.5
            print image[:2]
            plt.figure(figsize=(9,1))    
            for k in range(1, 8, 1):
                imgIdx = k
                graphIdx = ''.join(str(x) for x in [1,8,k])
                plt.subplot(graphIdx)
                plt.axis('off')
                plt.imshow(np.reshape(image[imgIdx], [28,28]), cmap="gray")
            
            plt.show()

eval()
