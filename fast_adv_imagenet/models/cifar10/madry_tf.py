# Adapted from:
# https://github.com/MadryLab/cifar10_challenge/blob/master/model.py
#
# based on https://github.com/tensorflow/models/tree/master/resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class Model(object):
    """ResNet model."""

    def __init__(self, mode):
        """ResNet constructor.

        Args:
          mode: One of 'train' and 'eval'.
        """
        self.mode = mode
        self.image_size = 32
        self.num_channels = 3
        self.num_labels = 10

        self.built = False

    def predict(self, input: tf.placeholder):
        out = self._build_model(input)
        self.built = True
        return out

    def add_internal_summaries(self):
        pass

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model(self, input):
        assert self.mode == 'train' or self.mode == 'eval'
        """Build the core model within the graph."""
        with tf.variable_scope('input', reuse=self.built):

            input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                                           input)
            x = self._conv('init_conv', input_standardized, 3, 3, 16, self._stride_arr(1))

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        res_func = self._residual

        # Uncomment the following codes to use w28-10 wide residual network.
        # It is more memory efficient than very deep residual network and has
        # comparably good performance.
        # https://arxiv.org/pdf/1605.07146v1.pdf
        filters = [16, 160, 320, 640]


        # Update hps.num_residual_units to 9

        with tf.variable_scope('unit_1_0', reuse=self.built):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                         activate_before_residual[0])
        for i in range(1, 5):
            with tf.variable_scope('unit_1_%d' % i, reuse=self.built):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

        with tf.variable_scope('unit_2_0', reuse=self.built):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                         activate_before_residual[1])
        for i in range(1, 5):
            with tf.variable_scope('unit_2_%d' % i, reuse=self.built):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

        with tf.variable_scope('unit_3_0', reuse=self.built):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                         activate_before_residual[2])
        for i in range(1, 5):
            with tf.variable_scope('unit_3_%d' % i, reuse=self.built):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

        with tf.variable_scope('unit_last', reuse=self.built):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, 0.1)
            x = self._global_avg_pool(x)

        with tf.variable_scope('logit', reuse=self.built):
            self.pre_softmax = self._fully_connected(x, 10)

        return self.pre_softmax

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.name_scope(name):
            return tf.contrib.layers.batch_norm(
                inputs=x,
                decay=.9,
                center=True,
                scale=True,
                activation_fn=None,
                updates_collections=None,
                is_training=(self.mode == 'train'))

    def _residual(self, x, in_filter, out_filter, stride,
                  activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation', reuse=self.built):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation', reuse=self.built):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)

        with tf.variable_scope('sub1', reuse=self.built):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2', reuse=self.built):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, 0.1)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add', reuse=self.built):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find('DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        return tf.add_n(costs)

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name, reuse=self.built):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0/n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        num_non_batch_dimensions = len(x.shape)
        prod_non_batch_dimensions = 1
        for ii in range(num_non_batch_dimensions - 1):
            prod_non_batch_dimensions *= int(x.shape[ii + 1])
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        w = tf.get_variable(
            'DW', [prod_non_batch_dimensions, out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])
