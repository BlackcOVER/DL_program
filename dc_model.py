import tensorflow as tf
import numpy as np
import os
from config import *
from utils import *


def generator(fake_input, reuse=False, label=None):
    if label:
       with tf.variable_scope('G_net', reuse=reuse):
           label_ = tf.reshape(label, [FLAGS.batch_size, 1, 1, 10]) 
           fake_input = tf.concat([fake_input, label], 1)

           layer1 = full_connect(fake_input, 'layer1', 110, 1024)
           layer1 = tf.nn.relu(tf.layers.batch_normalization(layer1))
           layer1 = tf.concat([layer1, label], 1)
 
           layer2 = full_connect(layer1, 'layer2', 1034, 128*7*7) 
           layer2 = tf.nn.relu(tf.layers.batch_normalization(layer2))
           layer2 = tf.reshape(layer2, [FLAGS.batch_size, 7, 7, 128])
           layer2 = tf.concat([layer2, label_*tf.ones([FLAGS.batch_size, 7, 7, 10])], 3)

           layer3 = deconv2d(layer2, 'layer3', 138, [FLAGS.batch_size, 14, 14, 128])
           layer3 = tf.nn.relu(tf.layers.batch_normalization(layer3))
           layer3 = tf.concat([layer3, label_*tf.ones([FLAGS.batch_size, 14, 14, 10])], 3)
       
           output = tf.nn.tanh(deconv2d(layer3, 'out_put', 138, [FLAGS.batch_size, 28, 28, 1]))
           #output = tf.nn.sigmoid(deconv2d(layer3, 'out_put', 138, [FLAGS.batch_size, 28, 28, 1]))
           return output 
    else:
        with tf.variable_scope('G_net', reuse=reuse):
           layer1 = full_connect(fake_input, 'layer1', 100, 1024)
           layer1 = tf.nn.relu(tf.layers.batch_normalization(layer1))

           layer2 = full_connect(layer1, 'layer2', 1024, 128*7*7)
           layer2 = tf.nn.relu(tf.layers.batch_normalization(layer2))
           layer2 = tf.reshape(layer2, [FLAGS.batch_size, 7, 7, 128])

           layer3 = deconv2d(layer2, 'layer3', 128, [FLAGS.batch_size, 14, 14, 128])
           layer3 = tf.nn.relu(tf.layers.batch_normalization(layer3))

           output = tf.nn.tanh(deconv2d(layer3, 'out_put', 128, [FLAGS.batch_size, 28, 28, 1]))
           #output = tf.nn.sigmoid(deconv2d(layer3, 'out_put', 128, [FLAGS.batch_size, 28, 28, 1]))
           return output

def discriminator(image, label=None, reuse=False):
    if label:
        with tf.variable_scope('D_net', reuse=reuse):
            label_ = tf.reshape(label, [FLAGS.batch_size, 1, 1, 10])
            input = tf.concat([image, label_*tf.ones([FLAGS.batch_size, 28, 28, 10])], 3)

            layer1 = conv2(input, 'layer1', 11, 11, reuse=False)
            layer1 = tf.nn.leaky_relu(layer1, alpha=0.2)
            layer1 = tf.concat([layer1, label_*tf.ones([FLAGS.batch_size, 14, 14, 10])], 3)

            layer2 = conv2(layer1, 'layer2', 21, 74, reuse=False) 
            layer2 = tf.nn.leaky_relu(tf.layers.batch_normalization(layer2), alpha=0.2) 
            layer2 = tf.reshape(layer2, [FLAGS.batch_size, -1])      
            layer2 = tf.concat([layer2, label], 1)
       
            layer3 = full_connect(layer2, 'layer3', 7*7*74+10, 1024)
            layer3 = tf.nn.leaky_relu(tf.layers.batch_normalization(layer3), alpha=0.2)
            layer3 = tf.concat([layer3, label], 1)

            layer4 = full_connect(layer3, 'layer4', 1024+10, 1)
        
            return tf.nn.sigmoid(layer4), layer4 
    else:
        with tf.variable_scope('D_net', reuse=reuse):
            layer1 = conv2(image, 'layer1', 1, 11, reuse=False)
            layer1 = tf.nn.leaky_relu(layer1, alpha=0.2)

            layer2 = conv2(layer1, 'layer2', 11, 74, reuse=False)
            layer2 = tf.nn.leaky_relu(tf.layers.batch_normalization(layer2), alpha=0.2)
            layer2 = tf.reshape(layer2, [FLAGS.batch_size, -1])

            layer3 = full_connect(layer2, 'layer3', 7*7*74, 1024)
            layer3 = tf.nn.leaky_relu(tf.layers.batch_normalization(layer3), alpha=0.2)

            layer4 = full_connect(layer3, 'layer4', 1024, 1)

            return tf.nn.sigmoid(layer4), layer4

def d_loss(logits_t, logits_f):
    #d_loss_t = tf.reduce_mean(
    #    tf.nn.sigmoid_cross_entropy_with_logits(
    #        logits=logits_t, labels=tf.ones_like(logits_t)))
    d_loss_t = tf.losses.sigmoid_cross_entropy(logits=logits_t, multi_class_labels=tf.ones_like(logits_t))
    #d_loss_f = tf.reduce_mean(
    #    tf.nn.sigmoid_cross_entropy_with_logits(
    #        logits=logits_f, labels=tf.zeros_like(logits_f)))
    d_loss_f = tf.losses.sigmoid_cross_entropy(logits=logits_f, multi_class_labels=tf.ones_like(logits_f))
    return d_loss_t + d_loss_f 

def g_loss(logits_f):
    #g_loss_f = tf.reduce_mean(
    #    tf.nn.sigmoid_cross_entropy_with_logits(
    #        logits=logits_f, labels=tf.ones_like(logits_f)))
    g_loss_f = tf.losses.sigmoid_cross_entropy(logits=logits_f, multi_class_labels=tf.ones_like(logits_f))
    return g_loss_f

def train(g_loss, d_loss):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):

        d_vars = [var for var in tf.trainable_variables() if 'D' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'G' in var.name]
        d_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
        d_train = d_opt.minimize(d_loss, var_list=d_vars)

        g_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
        g_train = g_opt.minimize(g_loss, var_list=g_vars)
        return  g_train, d_train
