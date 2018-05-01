import tensorflow as tf
from config import *
import math

def get_weight(name, shape, stddev):
    weight = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    return weight

def get_bias(name, shape):
    bias = tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))
    return bias 

def generator(fake_input, reuse, w_init):
    with tf.variable_scope('G_net', reuse=reuse), tf.device('/device:CPU:0'):
        layer = tf.layers.dense(fake_input, 1000, activation=tf.nn.relu, kernel_initializer=w_init)  
        layer = tf.layers.dense(layer, 1000, activation=tf.nn.relu, kernel_initializer=w_init)
        layer = tf.layers.batch_normalization(layer, training=True)
        logits = tf.layers.dense(layer, 784, kernel_initializer=w_init)
        outputs = tf.nn.tanh(logits)  
        #outputs = tf.nn.sigmoid(logits)
        return outputs

def discreminator(input, reuse, w_init):
    with tf.variable_scope('D_net', reuse=reuse), tf.device('/device:CPU:0'):
        layer = tf.layers.dense(input, 784, activation=tf.nn.leaky_relu, kernel_initializer=w_init)
        layer = tf.layers.dense(layer, 256, activation=tf.nn.leaky_relu, kernel_initializer=w_init)  
        layer = tf.layers.dense(layer, 64, activation=tf.nn.leaky_relu, kernel_initializer=w_init)
        logits = tf.layers.dense(layer, 1, kernel_initializer=w_init)  
        outputs = tf.nn.sigmoid(logits)  
        return outputs, logits  

def g_loss(fake_logits):
    g_loss_d = tf.losses.sigmoid_cross_entropy(logits=fake_logits, multi_class_labels=tf.ones_like(fake_logits) * (1-FLAGS.smooth))
    return g_loss_d

def d_loss(truth_logits, fake_logits):
    truth_loss_d = tf.losses.sigmoid_cross_entropy(logits=truth_logits, multi_class_labels=tf.ones_like(truth_logits) * (1-FLAGS.smooth))
    fake_loss_d = tf.losses.sigmoid_cross_entropy(logits=fake_logits, multi_class_labels=tf.zeros_like(fake_logits))
    total_loss = truth_loss_d + fake_loss_d 
    return total_loss

def train(g_loss, d_loss, g_learning_rate, d_learning_rate):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):

        d_vars = [var for var in tf.trainable_variables() if 'D' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'G' in var.name]

        # optimize D
        d_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
        d_train = d_opt.minimize(d_loss, var_list=d_vars)

        # optimize G
        g_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
        g_train = g_opt.minimize(g_loss, var_list=g_vars)
        return  g_train, d_train
    #g_net_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G_net')
    #train_g_op = tf.train.AdamOptimizer(g_learning_rate, beta1=0.5).minimize(g_loss, var_list=g_net_var_list)
    #d_net_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D_net')
    #train_d_op = tf.train.AdamOptimizer(d_learning_rate, beta1=0.5).minimize(d_loss, var_list=d_net_var_list)
    #train_d_op = tf.train.MomentumOptimizer(g_learning_rate, 0.9).minimize(d_loss, var_list=d_net_var_list)
    #return train_g_op, train_d_op
