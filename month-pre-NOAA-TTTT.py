#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽 TensorFlow 的 INFO/WARNING 日志
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'  # Graphviz 路径

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # 屏蔽 TensorFlow v1 日志
tf.config.threading.set_inter_op_parallelism_threads(4)  # 可选：控制并行线程数

# ==================== 基础库导入 ====================
import numpy as np
import pandas as pd
import math
from datetime import datetime
import csv

# ==================== 深度学习框架 (TensorFlow/Keras) ====================
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv3D, MaxPool3D, Concatenate, BatchNormalization,
    Activation, ConvLSTM2D, ReLU, Dense, Flatten, Dropout, Multiply, Permute
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.regularizers import l2

# 自定义 RNN 层所需底层模块（保持原样引用）
from tensorflow.python.keras import activations, backend, constraints, initializers, regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, RNN
from tensorflow.python.keras.utils import conv_utils, generic_utils, tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.layers.convolutional_recurrent import ConvRNN2D

# ==================== 科学计算与可视化 ====================
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.gridliner as gridliner

from netCDF4 import Dataset
from scipy.stats import pearsonr

# 设置随机种子以确保实验可重现
np.random.seed(0)
tf.random.set_seed(0)


class ConvGRU2DCell(DropoutRNNCellMixin, Layer):
    """Cell class for the ConvGRU2DCell layer.

    Args:
      filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        dimensions of the convolution window.
      strides: An integer or tuple/list of n integers,
        specifying the strides of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
        `"valid"` means no padding. `"same"` results in padding evenly to
        the left/right or up/down of the input such that output has the same
        height/width dimension as the input.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
        for the recurrent step.
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.

    Call arguments:
      inputs: A 4D tensor.
      states:  List of state tensors corresponding to the previous timestep.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(ConvGRU2DCell, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
                                                        'dilation_rate')
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.filters)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters * 3)
        self.kernel_shape = kernel_shape
        recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 3)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.filters * 3,),
                name='bias',
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state

        # dropout matrices for input units
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
        # dropout matrices for recurrent units
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=3)

        if 0 < self.dropout < 1.:
            inputs_z = inputs * dp_mask[0]
            inputs_r = inputs * dp_mask[1]
            inputs_h = inputs * dp_mask[2]

        else:
            inputs_z = inputs
            inputs_r = inputs
            inputs_h = inputs

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_z = h_tm1 * rec_dp_mask[0]
            h_tm1_r = h_tm1 * rec_dp_mask[1]
            h_tm1_h = h_tm1 * rec_dp_mask[2]

        else:
            h_tm1_z = h_tm1
            h_tm1_r = h_tm1
            h_tm1_h = h_tm1

        (kernel_z, kernel_r,
         kernel_h) = array_ops.split(self.kernel, 3, axis=3)
        (recurrent_kernel_z,
         recurrent_kernel_r,
         recurrent_kernel_h) = array_ops.split(self.recurrent_kernel, 3, axis=3)

        if self.use_bias:
            bias_z, bias_r, bias_h = array_ops.split(self.bias, 3)
        else:
            bias_z, bias_r, bias_h = None, None, None

        x_z = self.input_conv(inputs_z, kernel_z, bias_z, padding=self.padding)
        x_r = self.input_conv(inputs_r, kernel_r, bias_r, padding=self.padding)
        x_h = self.input_conv(inputs_h, kernel_h, bias_h, padding=self.padding)

        h_z = self.recurrent_conv(h_tm1_z, recurrent_kernel_z)
        h_r = self.recurrent_conv(h_tm1_r, recurrent_kernel_r)
        h_h = self.recurrent_conv(h_tm1_h, recurrent_kernel_h)

        z = self.recurrent_activation(x_z + h_z)
        r = self.recurrent_activation(x_r + h_r)

        h = (1.0 - z) * h_tm1 + z * self.activation(x_h + h_h)
        return h, [h]

    def input_conv(self, x, w, b=None, padding='valid'):
        conv_out = backend.conv2d(x, w, strides=self.strides,
                                  padding=padding,
                                  data_format=self.data_format,
                                  dilation_rate=self.dilation_rate)
        if b is not None:
            conv_out = backend.bias_add(conv_out, b,
                                        data_format=self.data_format)
        return conv_out

    def recurrent_conv(self, x, w):
        conv_out = backend.conv2d(x, w, strides=(1, 1),
                                  padding='same',
                                  data_format=self.data_format)
        return conv_out

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(
                      self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(
                      self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(
                      self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(
                      self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvGRU2DCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class ConvGRU2D(ConvRNN2D):
    """2D Convolutional GRU layer.

    A convolutional GRU is similar to an GRU, but the input transformations
    and recurrent transformations are both convolutional. This layer is typically
    used to process timeseries of images (i.e. video-like data).

    It is known to perform well for weather data forecasting,
    using inputs that are timeseries of 2D grids of sensor values.
    It isn't usually applied to regular video data, due to its high computational
    cost.

    Args:
      filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        dimensions of the convolution window.
      strides: An integer or tuple/list of n integers,
        specifying the strides of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
        `"valid"` means no padding. `"same"` results in padding evenly to
        the left/right or up/down of the input such that output has the same
        height/width dimension as the input.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, time, ..., channels)`
        while `channels_first` corresponds to
        inputs with shape `(batch, time, channels, ...)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      activation: Activation function to use.
        By default hyperbolic tangent activation function is applied
        (`tanh(x)`).
      recurrent_activation: Activation function to use
        for the recurrent step.
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to.
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence. (default False)
      return_state: Boolean Whether to return the last state
        in addition to the output. (default False)
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.

    Call arguments:
      inputs: A 5D float tensor (see input shape description below).
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or `recurrent_dropout`
        are set.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.

    Input shape:
      - If data_format='channels_first'
          5D tensor with shape:
          `(samples, time, channels, rows, cols)`
      - If data_format='channels_last'
          5D tensor with shape:
          `(samples, time, rows, cols, channels)`

    Output shape:
      - If `return_state`: a list of tensors. The first tensor is
        the output. The remaining tensors are the last states,
        each 4D tensor with shape:
        `(samples, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
      - If `return_sequences`: 5D tensor with shape:
        `(samples, timesteps, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 5D tensor with shape:
        `(samples, timesteps, new_rows, new_cols, filters)`
        if data_format='channels_last'.
      - Else, 4D tensor with shape:
        `(samples, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)`
        if data_format='channels_last'.

    Raises:
      ValueError: in case of invalid constructor arguments.

    References:
      - [Shi et al., 2015](http://arxiv.org/abs/1506.04214v1)
      (the current implementation does not include the feedback loop on the
      cells output).

    Example:

    ```python
    steps = 10
    height = 32
    width = 32
    input_channels = 3
    output_channels = 6

    inputs = tf.keras.Input(shape=(steps, height, width, input_channels))
    layer = ConvGRU2D.ConvGRU2D(filters=output_channels, kernel_size=3)
    outputs = layer(inputs)
    ```
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        cell = ConvGRU2DCell(filters=filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=padding,
                             data_format=data_format,
                             dilation_rate=dilation_rate,
                             activation=activation,
                             recurrent_activation=recurrent_activation,
                             use_bias=use_bias,
                             kernel_initializer=kernel_initializer,
                             recurrent_initializer=recurrent_initializer,
                             bias_initializer=bias_initializer,
                             kernel_regularizer=kernel_regularizer,
                             recurrent_regularizer=recurrent_regularizer,
                             bias_regularizer=bias_regularizer,
                             kernel_constraint=kernel_constraint,
                             recurrent_constraint=recurrent_constraint,
                             bias_constraint=bias_constraint,
                             dropout=dropout,
                             recurrent_dropout=recurrent_dropout,
                             dtype=kwargs.get('dtype'))
        super(ConvGRU2D, self).__init__(cell,
                                        return_sequences=return_sequences,
                                        return_state=return_state,
                                        go_backwards=go_backwards,
                                        stateful=stateful,
                                        **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super(ConvGRU2D, self).call(inputs,
                                           mask=mask,
                                           training=training,
                                           initial_state=initial_state)

    @property
    def filters(self):
        return self.cell.filters

    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(
                      self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(
                      self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(
                      self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(
                      self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvGRU2D, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
class InceptionBlk(Model):
    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()
        self.ch = ch
        self.strides = strides
        self.c1 = ConvBNRelu(2 * ch, kernelsz=1, strides=strides)
        self.c2_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_2 = ConvBNRelu(2 * ch, kernelsz=3, strides=1)
        self.c3_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c3_2 = ConvBNRelu(2 * ch, kernelsz=3, strides=1)
        self.c3_3 = ConvBNRelu(2 * ch, kernelsz=3, strides=1)

        self.c4_2 = ConvBNRelu(2 * ch, kernelsz=3, strides=strides)

    def call(self, x):
        x1 = self.c1(x)
        print("X1:", x1.shape)
        x2_1 = self.c2_1(x)
        print("X2_1:", x2_1.shape)
        x2_2 = self.c2_2(x2_1)
        print("X2_2:", x2_2.shape)
        x3_1 = self.c3_1(x)
        x3_2 = self.c3_2(x3_1)
        x3_3 = self.c3_3(x3_2)
        print("X3_3:", x3_3.shape)

        x4_2 = self.c4_2(x)
        print("X4_2:", x4_2.shape)
        # concat along axis=channel
        x = tf.concat([x1, x2_2, x3_3, x4_2], axis=-1)

        return x

class GAMAttention(layers.Layer):
    def __init__(self, in_channels, rate=4, position_encoding_dim=None, **kwargs):
        super(GAMAttention, self).__init__(**kwargs)

        self.position_encoding_dim = position_encoding_dim or in_channels

        # Channel Attention
        self.channel_attention = tf.keras.Sequential([
            layers.Dense(int(in_channels / rate), activation='relu'),
            layers.Dense(in_channels, activation='sigmoid')
        ])

        # Spatial Attention
        self.spatial_attention = tf.keras.Sequential([
            layers.Conv2D(int(in_channels / rate), kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(1, kernel_size=3, padding='same'),  # 输出通道数应为1
            layers.BatchNormalization(),
            layers.Activation('sigmoid')
        ])

    def positional_encoding(self, height, width, depth):
        depth = depth // 2
        height = tf.cast(height, tf.float32)
        width = tf.cast(width, tf.float32)
        positions_h = tf.range(height, dtype=tf.float32)[:, tf.newaxis, tf.newaxis]  # (height, 1, 1)
        positions_w = tf.range(width, dtype=tf.float32)[tf.newaxis, :, tf.newaxis]  # (1, width, 1)

        depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, tf.newaxis, :] / depth  # (1, 1, depth)

        angle_rates_h = 1 / (10000 ** depths)  # (1, 1, depth)
        angle_rates_w = angle_rates_h  # 可以使用相同的频率，也可以独立设置

        angle_rads_h = positions_h * angle_rates_h  # (height, 1, depth)
        angle_rads_w = positions_w * angle_rates_w  # (1, width, depth)

        pos_encoding_h = tf.concat([tf.sin(angle_rads_h), tf.cos(angle_rads_h)], axis=-1)  # (height, 1, depth)
        pos_encoding_w = tf.concat([tf.sin(angle_rads_w), tf.cos(angle_rads_w)], axis=-1)  # (1, width, depth)

        pos_encoding = pos_encoding_h + pos_encoding_w  # (height, width, depth)

        return pos_encoding

    def call(self, inputs):
        b, t, h, w, c = tf.unstack(tf.shape(inputs, out_type=tf.int32))
        x = inputs

        # Positional Encoding
        pos_encoding = self.positional_encoding(h, w, self.position_encoding_dim)
        print(f"Shape of pos_encoding after encoding: {pos_encoding.shape}")

        pos_encoding = tf.expand_dims(pos_encoding, axis=0)  # Add batch dimension
        pos_encoding = tf.expand_dims(pos_encoding, axis=1)  # Add time step dimension
        print(f"Shape of pos_encoding after adding batch and time step dims: {pos_encoding.shape}")

        # Broadcasting
        pos_encoding = tf.tile(pos_encoding, [b, t, 1, 1, 1])  # Broadcast to match input shape
        print(f"Shape of pos_encoding after broadcasting: {pos_encoding.shape}")

        x = x + pos_encoding

        # Channel Attention
        x_reshaped = tf.reshape(x, [-1, c])
        x_att = self.channel_attention(x_reshaped)
        x_att = tf.reshape(x_att, [b, t, h, w, c])
        x_channel_att = x * x_att

        # Spatial Attention
        x_spatial_att = self.spatial_attention(tf.reshape(x_channel_att, [b * t, h, w, c]))
        if len(x_spatial_att.shape) == 4:
            x_spatial_att = tf.expand_dims(x_spatial_att, axis=-1)
        x_spatial_att = tf.reshape(x_spatial_att, [b, t, h, w, 1])  # 恢复原始形状
        # Apply Spatial Attention
        out = x_channel_att * x_spatial_att
        # Residual Connection
        out = out + inputs

        return out

class ConvBNRelu(Model):
    def __init__(self, ch, kernelsz=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.model = tf.keras.models.Sequential([
            Conv3D(ch, kernelsz, strides=strides, padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, x):
        x = self.model(x, training=False)
        return x

class sstModel(tf.keras.Model):
    def __init__(self):
        super(sstModel, self).__init__()
        self.input_layer = Input(shape=(10, 40, 93, 73, 1))
        self.inception = InceptionBlk(12)
        self.att = GAMAttention(120)
        # 在这里为ConvGRU2D层添加L2正则化
        self.ConvLSTM2D0 = ConvGRU2D(filters=24, kernel_size=3, strides=1, padding='same',
                                     return_sequences=True)
        self.ConvLSTM2D = ConvGRU2D(filters=12, kernel_size=3, strides=1, padding='same',
                                    return_sequences=False)
        self.conv2 = Conv2D(filters=1, kernel_size=1, strides=1, padding='same')

    def call(self, x):
        print(f"x.shape: {x.shape}")
        x1_1 = self.ConvLSTM2D0(x)
        print(f"x.shape: {x.shape}")

        x1 = self.inception(x)
        print(f"x1.shape: {x1.shape}")
        x1 = tf.concat([x1_1, x1], axis=-1)

        x2 = self.att(x1)
        print("x2 Shape:", x2.shape)
        x3 = self.ConvLSTM2D(x2)
        print("x3 Shape:", x3.shape)
        y = self.conv2(x3)
        print("y Shape:", y.shape)
        return y

import xarray as xr

def my_normal(var, time, train_len, val_len, test_len):
    """
    对数据进行归一化处理，并返回分割后的训练集、验证集、测试集及其归一化参数。
    """
    train_data = var[:train_len, :, :]
    val_data = var[train_len:train_len + val_len, :, :]
    test_data = var[train_len + val_len:, :, :]

    max_data = np.nanmax(train_data)
    min_data = np.nanmin(train_data)

    train_data_scale = (train_data - min_data) / (max_data - min_data)
    val_data_scale = (val_data - min_data) / (max_data - min_data)
    test_data_scale = (test_data - min_data) / (max_data - min_data)

    return train_data_scale, val_data_scale, test_data_scale, max_data, min_data
def loadData(rate=0.143, window_size=24):
    path_sst = r'D:\zyh\dataset\OISST\oisst_monthly_average_1981_2023.nc'

    # 使用 xarray 打开文件并读取数据
    ds_sst = xr.open_dataset(path_sst)

    target_lon_range = (103.125, 120.875)  # 目标经度范围
    target_lat_range = (0.125, 22.875)  # 目标纬度范围

    # 提取目标区域内的数据
    sst = ds_sst['sst'].sel(lon=slice(target_lon_range[0], target_lon_range[1]),
                            lat=slice(target_lat_range[0], target_lat_range[1]))
    sst = sst.squeeze()  # 这将移除所有长度为1的维度
    print(f"sst shape after slicing: {sst.shape}")
    lon = sst['lon'].values
    lat = sst['lat'].values

    sst = sst.where(sst.notnull(), np.nan)
    np.nan_to_num(sst.values, copy=False, nan=0.0)

    ds_sst.close()
    time = sst.shape[0]

    test_len = int(time * rate)
    remaining = time - test_len
    val_len = int(remaining * 2 / 8)
    train_len = remaining - val_len

    print(f"Train Length: {train_len}, Validation Length: {val_len}, Test Length: {test_len}")

    train_sst_scale, val_sst_scale, test_sst_scale, max_sst, min_sst = my_normal(sst, time, train_len, val_len,
                                                                                 test_len)

    feature = 1
    train_data = np.zeros((train_len, sst.shape[1], sst.shape[2], feature), dtype='float32')
    val_data = np.zeros((val_len, sst.shape[1], sst.shape[2], feature), dtype='float32')
    test_data = np.zeros((test_len, sst.shape[1], sst.shape[2], feature), dtype='float32')

    train_data[:, :, :, 0] = train_sst_scale
    val_data[:, :, :, 0] = val_sst_scale
    test_data[:, :, :, 0] = test_sst_scale

    # 应用滑动窗口逻辑到训练集、验证集和测试集
    x_train, y_train = [], []
    for i in range(window_size, train_data.shape[0]):
        x_train.append(train_data[i - window_size:i, :, :, :])
        y_train.append(train_data[i, :, :, 0])
    np.random.seed(12)
    np.random.shuffle(x_train)
    np.random.seed(12)
    np.random.shuffle(y_train)
    tf.random.set_seed(12)

    x_val, y_val = [], []
    for i in range(window_size, val_data.shape[0]):
        x_val.append(val_data[i - window_size:i, :, :, :])
        y_val.append(val_data[i, :, :, 0])

    x_test, y_test = [], []
    for i in range(window_size, test_data.shape[0]):
        x_test.append(test_data[i - window_size:i, :, :, :])
        y_test.append(test_data[i, :, :, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)[:, :, :, np.newaxis]
    x_val = np.array(x_val)
    y_val = np.array(y_val)[:, :, :, np.newaxis]
    x_test = np.array(x_test)
    y_test = np.array(y_test)[:, :, :, np.newaxis]

    return x_train, y_train, x_val, y_val, x_test, y_test, max_sst, min_sst, sst, lon, lat
def generator_data(inputs, target, batch_size):
    i = 0
    while 1:
        x_train = []
        y_train = []
        for _ in range(batch_size):
            if i == 0:
                state = np.random.get_state()
                np.random.shuffle(inputs)
                np.random.set_state(state)
                np.random.shuffle(target)
            x_train.append(inputs[i, :, :, :, :])
            y_train.append(target[i, :, :, :])
            i = (i + 1) % inputs.shape[0]
        yield (np.array(x_train), np.array(y_train))
def generator_data_test(inputs, target, batch_size):
    i = 0
    while True:
        x_train = []
        y_train = []
        for _ in range(batch_size):
            x_train.append(inputs[i, :, :, :, :])
            y_train.append(target[i, :, :, :])
            i = (i + 1) % inputs.shape[0]

        # 转换为 NumPy 数组
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # print("x_train shape:", x_train.shape)
        # print("y_train shape:", y_train.shape)

        yield (x_train, y_train)
def get_lr_metric(optimizer):  # printing the value of the learning rate
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr
def customized_mse(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    mse = tf.reduce_mean(squared_difference)
    return mse

def customized_mae(y_true, y_pred):
    absolute_difference = tf.abs(y_true - y_pred)
    mae = tf.reduce_mean(absolute_difference)
    return mae
def customized_rmse(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    rmse = tf.sqrt(mse)
    return rmse
def customized_r2(y_true, y_pred):
    mask_sea = ~np.equal(y_true[:, :, 0], 0)
    sea_num = np.sum(mask_sea)
    if sea_num == 0:
        sea_num = 1.0
    residual = np.sum(np.square(y_true - y_pred))
    total = np.sum(np.square(y_true - np.mean(y_true[mask_sea])))
    r_squared = 1 - (residual / total) if total != 0 else 0
    return r_squared

def predict_next_day(model, x_input):
    predicted_sst = model.predict(np.expand_dims(x_input, axis=0))
    return np.squeeze(predicted_sst, axis=0)

if __name__ == '__main__':

    dt_start = datetime.now()
    # 加载数据
    x_train, y_train, x_val, y_val, x_test, y_test, max_sst, min_sst, sst, lon, lat = loadData(0.143)
    print("Minimum Sea Surface Temperature:", min_sst)
    print("Maximum Sea Surface Temperature:", max_sst)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 根据需要更改GPU索引
    # 定义跨设备操作策略
    cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
    # 使用 MirroredStrategy 进行多 GPU 训练
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)

    with strategy.scope():
        # 创建模型实例
        model = sstModel()

        # 使用一些虚拟输入调用模型以构建模型结构
        dummy_input = np.zeros((4, 24, 92, 72, 1))  # 根据您的模型调整输入形状
        _ = model(dummy_input)

        # 加载预训练权重
        model.load_weights(f"D:\zyh\TTTT\weight\TTTT-mon\weights.h5")
    '''
if __name__ == '__main__':
    dt_start = datetime.now()
    # 设置可见的GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 根据需要更改GPU索引
    cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    BATCH_SIZE_PER_REPLICA = 4

    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    with strategy.scope():
        optimizer = Adam(learning_rate=0.001)
        lr_metric = get_lr_metric(optimizer)
        model = sstModel()
        model.build(input_shape=(GLOBAL_BATCH_SIZE, 24, 92, 72, 1))  # 使用全局批大小构建模型

        model.compile(optimizer=optimizer, loss=customized_mse, metrics=[lr_metric])

        x_train, y_train, x_val, y_val, x_test, y_test, max_sst, min_sst, sst, lon, lat = loadData(0.143)
        print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)
        print("Minimum Sea Surface Temperature:", min_sst)
        print("Maximum Sea Surface Temperature:", max_sst)
        filepath = f"D:\zyh\TTTT\weight\TTTT-mon\weights.h5"

        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=True, mode='min')
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, mode='min', verbose=1,
                                                         factor=0.5)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
        history = model.fit(generator_data(x_train, y_train, GLOBAL_BATCH_SIZE),
                            steps_per_epoch=max(1, x_train.shape[0] // GLOBAL_BATCH_SIZE), epochs=200,
                            validation_data=generator_data_test(x_test, y_test, GLOBAL_BATCH_SIZE),
                            validation_steps=max(1, x_test.shape[0] // GLOBAL_BATCH_SIZE), initial_epoch=0,
                            callbacks=[checkpoint, reduce_lr, early_stopping])

        model.summary()
        loss = np.array(history.history['loss']).astype(np.float32) * pow((max_sst - min_sst), 2)
        val_loss = np.array(history.history["val_loss"]).astype(np.float32) * pow((max_sst - min_sst), 2)

        np.savetxt(f'D:\zyh\CNN+LSTM im\loss\loss.txt', loss, delimiter=',', fmt='%.6f')
        print('Training loss saved!')
        np.savetxt(f'D:\zyh\CNN+LSTM im\loss\\val_loss.txt', val_loss, delimiter=',', fmt='%.6f')
        print('Validation loss saved!')

        train_loss = np.loadtxt(f'D:\zyh\CNN+LSTM im\loss\loss.txt', delimiter=',')
        val_loss = np.loadtxt(f'D:\zyh\CNN+LSTM im\loss\\val_loss.txt', delimiter=',')

        import matplotlib.pyplot as plt

        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(f'./images/loss_plot.png')
    '''

'''
num_days_to_predict = 12
grid_shape = (92, 72)

# 初始化矩阵用于存储每个网格点的累积RMSE和MAE
cumulative_rmse_per_grid = np.zeros(grid_shape)
cumulative_mae_per_grid = np.zeros(grid_shape)

for i in range(x_test.shape[0] - num_days_to_predict):
    print(f"当前索引i: {i}")
    current_input = x_test[i]
    predictions = np.zeros((num_days_to_predict, 92, 72))
    for day in range(num_days_to_predict):
        predicted_sst = predict_next_day(model, current_input)

        current_input = np.concatenate((current_input[1:], predicted_sst[np.newaxis, :, :, :]), axis=0)

        predictions[day] = np.squeeze(predicted_sst)

    predicted_sst_denorm = np.squeeze(predictions[11] * (max_sst - min_sst) + min_sst)
    true_sst_denorm = np.squeeze(y_test[i + num_days_to_predict - 1] * (max_sst - min_sst) + min_sst)
    lower_threshold = 12.43793
    upper_threshold = 31.477098
    # 对预测的SST应用双向掩码：小于lower_threshold和大于upper_threshold的值都不绘制
    predicted_sst_denorm = np.ma.masked_outside(predicted_sst_denorm[:, :], lower_threshold,
                                                upper_threshold)

    true_sst_denorm = np.ma.masked_outside(true_sst_denorm[:, :], lower_threshold, upper_threshold)

    rmse_map = np.square(true_sst_denorm - predicted_sst_denorm)
    mae_map = np.abs(true_sst_denorm - predicted_sst_denorm)

    cumulative_rmse_per_grid += rmse_map
    cumulative_mae_per_grid += mae_map
average_rmse_per_grid = np.sqrt(cumulative_rmse_per_grid / (len(x_test) - num_days_to_predict + 1))
average_mae_per_grid = cumulative_mae_per_grid / (len(x_test) - num_days_to_predict + 1)

average_rmse_per_grid = np.ma.masked_equal(average_rmse_per_grid, 0)
average_mae_per_grid = np.ma.masked_equal(average_mae_per_grid, 0)

mean_average_rmse = average_rmse_per_grid.mean()
mean_average_mae = average_mae_per_grid.mean()

# 定义地理坐标范围
lat_min, lat_max = 0.125, 22.875
lon_min, lon_max = 103.125, 120.875
# 找到对应的索引范围
lat_indices = np.where((lat >= lat_min) & (lat <= lat_max))
lon_indices = np.where((lon >= lon_min) & (lon <= lon_max))
region_average_rmse_per_grid = average_rmse_per_grid[lat_indices[0][:, None], lon_indices[0]]
region_average_mae_per_grid = average_mae_per_grid[lat_indices[0][:, None], lon_indices[0]]
# 计算该区域的平均值和标准差
region_rmse_mean = np.mean(region_average_rmse_per_grid)
region_rmse_std = np.std(region_average_rmse_per_grid)
region_mae_mean = np.mean(region_average_mae_per_grid)
region_mae_std = np.std(region_average_mae_per_grid)
print(f"Region RMSE Mean: {region_rmse_mean}")
print(f"Region RMSE Std: {region_rmse_std}")
print(f"Region MAE Mean: {region_mae_mean}")
print(f"Region MAE Std: {region_mae_std}")

average_rmse_per_grid[lat_indices[0][:, None], lon_indices[0]] = region_average_rmse_per_grid
average_mae_per_grid[lat_indices[0][:, None], lon_indices[0]] = region_average_mae_per_grid

print("特定地理区域的平均RMSE: ", region_rmse_mean)
print("特定地理区域的标准差(RMSE): ", region_rmse_std)
print("特定地理区域内的最小Average RMSE: ", region_average_rmse_per_grid.min())
print("特定地理区域内的最大Average RMSE: ", region_average_rmse_per_grid.max())
print("特定地理区域的平均MAE: ", region_mae_mean)
print("特定地理区域的标准差(MAE): ", region_mae_std)
print("特定地理区域内的最小Average MAE: ", region_average_mae_per_grid.min())
print("特定地理区域内的最大Average MAE: ", region_average_mae_per_grid.max())
print("Minimum Average RMSE:", average_rmse_per_grid.min())
print("Maximum Average RMSE:", average_rmse_per_grid.max())
print("Minimum Average MAE:", average_mae_per_grid.min())
print("Maximum Average MAE:", average_mae_per_grid.max())

directory = 'month-TTTT-12'
if not os.path.exists(directory):
    os.makedirs(directory)

plt.figure(figsize=(4, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
im = ax.imshow(average_rmse_per_grid, transform=ccrs.PlateCarree(), cmap='jet', interpolation='bilinear',
               extent=[lon.min(), lon.max(), lat.min(), lat.max()], vmin=0.218, vmax=3.5,
               origin='lower')  # 添加此行以修正图像方向

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xformatter = gridliner.LongitudeFormatter()
gl.yformatter = gridliner.LatitudeFormatter()
gl.xlabel_style = {'size': 16, 'color': 'black', 'family': 'serif', 'fontname': 'Times New Roman'}
gl.ylabel_style = {'size': 16, 'color': 'black', 'family': 'serif', 'fontname': 'Times New Roman'}
# 添加颜色条并设置字体
cbar = plt.colorbar(im, shrink=0.7)
cbar.ax.tick_params(labelsize=14)  # 设置颜色条字体大小
# 设置颜色条标签格式为保留两位小数的新罗马字体
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_family('Times New Roman')
    l.set_size(14)  # 确保字体大小一致
    l.set_weight('normal')  # 字体粗细
    l.set_rotation(0)  # 标签旋转角度

# 格式化颜色条刻度值为保留两位小数
cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
plt.tight_layout()  # 确保子图之间不会重叠
plt.savefig(os.path.join(directory, 'TTTT-Average_RMSE_12th_month.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)  # 保存图像
plt.close()

# 绘制平均MAE
plt.figure(figsize=(4, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
im = ax.imshow(average_mae_per_grid, transform=ccrs.PlateCarree(), cmap='jet', interpolation='bilinear',
               extent=[lon.min(), lon.max(), lat.min(), lat.max()], vmin=0.185, vmax=2.5,
               origin='lower')  # 添加此行以修正图像方向

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xformatter = gridliner.LongitudeFormatter()
gl.yformatter = gridliner.LatitudeFormatter()
gl.xlabel_style = {'size': 16, 'color': 'black', 'family': 'serif', 'fontname': 'Times New Roman'}
gl.ylabel_style = {'size': 16, 'color': 'black', 'family': 'serif', 'fontname': 'Times New Roman'}
# 添加颜色条并设置字体
cbar = plt.colorbar(im, shrink=0.7)
cbar.ax.tick_params(labelsize=14)  # 设置颜色条字体大小
# 设置颜色条标签格式为保留两位小数的新罗马字体
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_family('Times New Roman')
    l.set_size(14)  # 确保字体大小一致
    l.set_weight('normal')  # 字体粗细
    l.set_rotation(0)  # 标签旋转角度
# 格式化颜色条刻度值为保留两位小数
cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

plt.tight_layout()  # 确保子图之间不会重叠
plt.savefig(os.path.join(directory, 'TTTT-Average_MAE_12th_month.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)  # 保存图像
plt.close()  # 关闭图像


dt_end = datetime.now()
print("Total time taken:", dt_end - dt_start)
'''

'''
num_months_to_predict = 12
# 初始化列表用于存储每天的评价指标和预测值
rmse_list, mae_list, r2_list, corr_list = [[] for _ in range(num_months_to_predict)], \
                                          [[] for _ in range(num_months_to_predict)], \
                                          [[] for _ in range(num_months_to_predict)], \
                                          [[] for _ in range(num_months_to_predict)]
prediction_list = [[] for _ in range(num_months_to_predict)]  # 用于存储每天的预测值
# 初始化列表用于存储每天的最大和最小值
true_max_values, true_min_values = [[] for _ in range(num_months_to_predict)], \
                                   [[] for _ in range(num_months_to_predict)]
pred_max_values, pred_min_values = [[] for _ in range(num_months_to_predict)], \
                                   [[] for _ in range(num_months_to_predict)]
for i in range(x_test.shape[0] - num_months_to_predict):

    print(f"当前索引i: {i}")
    current_input = x_test[i]
    for month in range(num_months_to_predict):
        predicted_sst = predict_next_day(model, current_input)
        current_input = np.concatenate((current_input[1:], predicted_sst[np.newaxis, :, :, :]), axis=0)
        predicted_sst_denorm = predicted_sst * (max_sst - min_sst) + min_sst
        y_test_denorm = y_test[i + month] * (max_sst - min_sst) + min_sst

        prediction_list[month].append(predicted_sst_denorm)  # 保存原始形状的预测值
        # 记录真实值和预测值的最大和最小值
        true_max_values[month].append(np.max(y_test_denorm))
        true_min_values[month].append(np.min(y_test_denorm))
        pred_max_values[month].append(np.max(predicted_sst_denorm))
        pred_min_values[month].append(np.min(predicted_sst_denorm))

# 打印每一天的最大和最小值
for month in range(num_months_to_predict):
    print(f"Day {month + 1}:")
    print(f"True Values Max: {np.max(true_max_values[month]):.2f}, Min: {np.min(true_min_values[month]):.2f}")
    print(f"Predicted Values Max: {np.max(pred_max_values[month]):.2f}, Min: {np.min(pred_min_values[month]):.2f}")

for month in range(num_months_to_predict):
    fig, ax = plt.subplots(figsize=(5, 5))  # 创建一个新的图形对象

    # 将预测值和真实值展平为一维数组
    true_sst_flatten = np.concatenate(
        [y_test[i + month] * (max_sst - min_sst) + min_sst for i in range(x_test.shape[0] - num_months_to_predict + 1)])
    pred_sst_flatten = np.concatenate(prediction_list[month])

    # 只考虑非陆地区域
    mask_sea = ~np.equal(true_sst_flatten, 0)

    # 打印调整前的长度
    print(
        f"Before adjustment: mask_sea={len(mask_sea)}, true_sst_flatten={len(true_sst_flatten)}, pred_sst_flatten={len(pred_sst_flatten)}")

    # 检查并调整长度以确保所有数组具有相同的长度
    if len(mask_sea) != len(true_sst_flatten) or len(mask_sea) != len(pred_sst_flatten):
        # 找到最小长度
        min_length = min(len(mask_sea), len(true_sst_flatten), len(pred_sst_flatten))

        # 调整各数组的长度
        mask_sea = mask_sea[:min_length]
        true_sst_flatten = true_sst_flatten[:min_length]
        pred_sst_flatten = pred_sst_flatten[:min_length]

    # 打印调整后的长度
    print(
        f"After adjustment: mask_sea={len(mask_sea)}, true_sst_flatten={len(true_sst_flatten)}, pred_sst_flatten={len(pred_sst_flatten)}")

    from matplotlib.font_manager import FontProperties

    #设置新罗马字体
    font = FontProperties(family='Times New Roman', size=16)

    cdict = {
        'red': [(0.0, 0.0, 0.0),  # 起点（深蓝色）无红色
                (0.2, 0.0, 0.0),
                (0.4, 0.0, 0.0),
                (0.6, 1.0, 1.0),
                (0.8, 1.0, 1.0),
                (0.9, 1.0, 1.0),
                (1.0, 0.6, 0.6)],  # 终点（深红色），稍微混合一些黑色

        'green': [(0.0, 0.0, 0.0),  # 深蓝色起点无绿色
                  (0.2, 0.4, 0.4),
                  (0.4, 0.8, 0.8),
                  (0.6, 1.0, 1.0),
                  (0.8, 0.5, 0.5),
                  (1.0, 0.0, 0.0)],

        'blue': [(0.0, 0.7, 0.7),  # 起点（深蓝色），稍微混合一些黑色
                 (0.2, 0.8, 0.8),
                 (0.3, 0.9, 0.9),
                 (0.6, 0.0, 0.0),
                 (0.8, 0.0, 0.0),
                 (1.0, 0.0, 0.0)]  # 终点无蓝色
    }

    cmap = mcolors.LinearSegmentedColormap('custom_cmap', cdict)
    # 使用 hexbin 绘制散点图，通过颜色表示密度
    hb = ax.hexbin(true_sst_flatten[mask_sea], pred_sst_flatten[mask_sea], gridsize=400, cmap=cmap,
                   bins='log',vmin=1,vmax=141)  # 确保颜色条的范围也对应于温度区间

    vmin_log, vmax_log = hb.get_clim()
    print(f"Log-scaled color bar range: Min = {vmin_log}, Max = {vmax_log}")
    # 添加颜色条
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Counts', fontproperties=font, fontsize=20)
    # 放大颜色条的刻度标签字体
    cb.ax.tick_params(labelsize=18)  # 设置颜色条刻度标签字体大小为16
    for label in cb.ax.get_yticklabels():
        label.set_fontproperties(font)  # 如果需要使用自定义字体
    # 设置子图标题
    ax.set_title(f'Month {month + 1}', fontproperties=font, fontsize=22)

    # 设置 X 轴和 Y 轴标签，并增大字体
    ax.set_ylabel('Predicted SST (°C)', fontproperties=font, fontsize=16)
    ax.set_xlabel('True SST (°C)', fontproperties=font, fontsize=16)

    # 固定横纵坐标范围为15-35°C
    ax.set_xlim(13, 33)
    ax.set_ylim(13, 33)

    # 设置横纵坐标的刻度间隔为5
    ax.set_xticks(range(13, 34, 5))  # 设置x轴刻度间隔为5
    ax.set_yticks(range(13, 34, 5))  # 设置y轴刻度间隔为5

    # 设置坐标轴刻度方向为朝内
    ax.tick_params(axis='both', which='both', direction='in')

    # 设置刻度标签字体
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)
        label.set_fontsize(16)  # 刻度标签字体大小设为14

    # 添加对角线（理想情况下，预测值应与真实值完全一致）
    lims = [13, 33]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)

    # 设置图形比例相等，以便对角线准确表示y=x
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # 调整布局
    plt.tight_layout()

    # 确保保存图表的文件夹存在，如果不存在则创建
    output_folder = 'TTTT-scatter_plots-12'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 保存图表到文件，为每一天指定不同的文件名
    filename = os.path.join(output_folder, f'TTTT_plot_monthly_{month + 1}.png')
    plt.savefig(filename, dpi=600, bbox_inches='tight')

    # 关闭当前图形以释放内存
    plt.close(fig)
dt_end = datetime.now()
print("Total time taken:", dt_end - dt_start)
'''

num_days_to_predict = 12

rmse_list, mae_list, r2_list, corr_list = [[] for _ in range(num_days_to_predict)], \
    [[] for _ in range(num_days_to_predict)], \
    [[] for _ in range(num_days_to_predict)], \
    [[] for _ in range(num_days_to_predict)]
prediction_list = [[] for _ in range(num_days_to_predict)]

true_max_values, true_min_values = [[] for _ in range(num_days_to_predict)], \
    [[] for _ in range(num_days_to_predict)]
pred_max_values, pred_min_values = [[] for _ in range(num_days_to_predict)], \
    [[] for _ in range(num_days_to_predict)]

# 滚动预测
for i in range(x_test.shape[0] - num_days_to_predict):
    print(f"当前索引 i: {i}")
    current_input = x_test[i].copy()  # (T, H, W, 1)

    for day in range(num_days_to_predict):
        predicted_sst = predict_next_day(model, current_input)  # (H, W, 1)
        # 滚动更新输入
        current_input = np.concatenate((current_input[1:], predicted_sst[np.newaxis, :, :, :]), axis=0)

        # 反归一化
        predicted_sst_denorm = predicted_sst * (max_sst - min_sst) + min_sst  # (H, W, 1)
        y_test_denorm = y_test[i + day] * (max_sst - min_sst) + min_sst  # (H, W, 1)

        # 存储预测值（保持形状）
        prediction_list[day].append(predicted_sst_denorm.squeeze())  # 去掉通道维，存为 (H, W)

        # 记录最大最小值
        true_max_values[day].append(np.max(y_test_denorm))
        true_min_values[day].append(np.min(y_test_denorm))
        pred_max_values[day].append(np.max(predicted_sst_denorm))
        pred_min_values[day].append(np.min(predicted_sst_denorm))

# 打印每日极值
for day in range(num_days_to_predict):
    print(f"Day {day + 1}:")
    print(f"  True Max: {np.max(true_max_values[day]):.2f}, Min: {np.min(true_min_values[day]):.2f}")
    print(f"  Pred Max: {np.max(pred_max_values[day]):.2f}, Min: {np.min(pred_min_values[day]):.2f}")

# 自定义颜色映射
cdict = {
    'red': [(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.4, 0.0, 0.0), (0.6, 1.0, 1.0), (0.8, 1.0, 1.0), (0.9, 1.0, 1.0),
            (1.0, 0.6, 0.6)],
    'green': [(0.0, 0.0, 0.0), (0.2, 0.4, 0.4), (0.4, 0.8, 0.8), (0.6, 1.0, 1.0), (0.8, 0.5, 0.5), (1.0, 0.0, 0.0)],
    'blue': [(0.0, 0.7, 0.7), (0.2, 0.8, 0.8), (0.3, 0.9, 0.9), (0.6, 0.0, 0.0), (0.8, 0.0, 0.0), (1.0, 0.0, 0.0)]
}
cmap = mcolors.LinearSegmentedColormap('custom_cmap', cdict)

# 输出文件夹
output_folder = 'month_TTTT-scatter_plots-30'
os.makedirs(output_folder, exist_ok=True)

# ===== 开始绘图并计算 R² =====
for day in range(num_days_to_predict):
    # 拼接所有样本的 true 和 pred
    true_sst_flatten = np.concatenate([y_test[i + day] * (max_sst - min_sst) + min_sst
                                       for i in range(x_test.shape[0] - num_days_to_predict)])
    true_sst_flatten = true_sst_flatten.squeeze()  # -> (N*H*W,)

    pred_sst_flatten = np.concatenate([p for p in prediction_list[day]])  # (N, H, W) -> (N*H*W,)

    # 创建海洋掩码（非陆地）
    mask_sea = (true_sst_flatten != 0)  # 假设陆地为 0

    # 应用掩码
    true_valid = true_sst_flatten[mask_sea]
    pred_valid = pred_sst_flatten[mask_sea]

    # ✅ 计算 R²
    r2 = r2_score(true_valid, pred_valid)
    corr, _ = pearsonr(true_valid, pred_valid)
    rmse = np.sqrt(mean_squared_error(true_valid, pred_valid))
    mae = mean_absolute_error(true_valid, pred_valid)

    # 保存指标
    r2_list[day] = r2
    corr_list[day] = corr
    rmse_list[day] = rmse
    mae_list[day] = mae

    # 绘图
    fig, ax = plt.subplots(figsize=(5, 5))

    hb = ax.hexbin(true_valid, pred_valid, gridsize=400, cmap=cmap, bins='log', vmin=1, vmax=1352)

    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Counts', fontsize=20, fontfamily='Times New Roman')
    cb.ax.tick_params(labelsize=18)
    for label in cb.ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')

    # ✅ 在图上添加 R², RMSE, MAE, Corr
    text_str = f'R² = {r2:.3f}\nRMSE = {rmse:.2f}°C\nMAE = {mae:.2f}°C\nr = {corr:.3f}'
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=14,
    verticalalignment = 'top', bbox = dict(boxstyle="round", facecolor="white", alpha=0.8),
    fontfamily = 'Times New Roman')

    ax.set_title(f'Day {day + 1}', fontsize=22, fontfamily='Times New Roman')
    ax.set_ylabel('Predicted SST (°C)', fontsize=16, fontfamily='Times New Roman')
    ax.set_xlabel('True SST (°C)', fontsize=16, fontfamily='Times New Roman')

    ax.set_xlim(15, 35)
    ax.set_ylim(15, 35)
    ax.set_xticks(range(15, 36, 5))
    ax.set_yticks(range(15, 36, 5))
    ax.tick_params(axis='both', which='both', direction='in', labelsize=16)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')

    ax.plot([15, 35], [15, 35], 'k--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')

    plt.tight_layout()
    filename = os.path.join(output_folder, f'month_TTTT_plot_day_{day + 1}.png')
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close(fig)

    print(f"Day {day + 1:2d} | R²={r2:.3f}, RMSE={rmse:.2f}°C, MAE={mae:.2f}°C, r={corr:.3f}")

    # 可选：保存指标到文件
    metrics = {
'day': list(range(1, 13)),
'R2': r2_list,
'RMSE': rmse_list,
'MAE': mae_list,
'Corr': corr_list
}
import pandas as pd

pd.DataFrame(metrics).to_csv(os.path.join(output_folder, 'evaluation_metrics_30days.csv'), index=False)

dt_end = datetime.now()
print("Total time taken:", dt_end - dt_start)
