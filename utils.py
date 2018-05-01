import tensorflow as tf
import pandas as pd

def get_weight(name, shape, stddev=0.02, trainable=True):
    weight = tf.get_variable(name, shape, tf.float32, trainable=trainable, initializer=tf.random_normal_initializer(stddev=stddev))
    return weight

def get_bias(name, shape, trainable=True):
    bias = tf.get_variable(name, shape, trainable=trainable, initializer=tf.constant_initializer(0.0))
    return bias

def one_hot(labels):
    return tf.one_hot(labels, 10, 1.0, 0.0)

def conv2(input, name, source_dim, dim, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        kernel = get_weight('weight', [5, 5, source_dim, dim])
        conv = tf.nn.conv2d(input, kernel, [1, 2, 2, 1], padding='SAME')
        bias = get_bias('bias', [dim])
        conv = tf.nn.bias_add(conv, bias)
        return conv

def full_connect(input, name, in_dim, out_dim, reuse=False, stddev=0.02):
    with tf.variable_scope(name, reuse=reuse):
        weight = get_weight('weight', [in_dim, out_dim])
        logits = tf.matmul(input, weight)
        bias = get_bias('bias', [out_dim])
        return tf.nn.bias_add(logits, bias)
 
def deconv2d(input, name, input_dim, output_shape, stddev=0.02, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        kernel = get_weight('weight', [5, 5, output_shape[-1], input_dim]) 
        deconv = tf.nn.conv2d_transpose(input, kernel, output_shape=output_shape, strides=[1, 2, 2, 1])
        biases = get_bias('bias', [output_shape[-1]])
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        return deconv
