#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import matplotlib

matplotlib.use('Agg')
from netCDF4 import Dataset
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Conv3D, Dropout, Conv2D, Conv3D, MaxPool3D, Concatenate, BatchNormalization, \
    Activation, ConvLSTM2D, ReLU, Dense, Layer, Multiply, Permute, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import pandas as pd
import math
from tensorflow.keras.losses import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'  # 这里是 Graphviz 的安装路径，请根据实际情况修改

from tensorflow.python.keras import activations, backend, constraints, initializers, regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, RNN
from tensorflow.python.keras.utils import conv_utils, generic_utils, tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.layers.convolutional_recurrent import ConvRNN2D
from tensorflow.keras.regularizers import l2

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
      - [Shi et al., 2015](http://arxiv.org/abs/1246.04214v1)
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
            layers.Conv2D(1, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('sigmoid')
        ])

    # ---------- 位置编码 ----------
    def positional_encoding(self, height, width, depth):
        depth = depth // 2
        height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
        pos_h = tf.range(height)[:, tf.newaxis, tf.newaxis]          # (H,1,1)
        pos_w = tf.range(width)[tf.newaxis, :, tf.newaxis]           # (1,W,1)
        depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, tf.newaxis, :] / depth  # (1,1,D)

        angle_rates = 1.0 / (10000.0 ** depths)
        angle_rads_h = pos_h * angle_rates                           # (H,1,D)
        angle_rads_w = pos_w * angle_rates                           # (1,W,D)

        pos_h = tf.concat([tf.sin(angle_rads_h), tf.cos(angle_rads_h)], axis=-1)  # (H,1,2D)
        pos_w = tf.concat([tf.sin(angle_rads_w), tf.cos(angle_rads_w)], axis=-1)  # (1,W,2D)
        pos_enc = pos_h + pos_w                                      # (H,W,2D)
        return pos_enc

    # ---------- 前向 ----------
    def call(self, inputs, return_attention=False):
        b, t, h, w, c = tf.unstack(tf.shape(inputs, out_type=tf.int32))
        x = inputs

        # 1) 位置编码
        pos_enc = self.positional_encoding(h, w, self.position_encoding_dim)   # (H,W,D)
        pos_enc = tf.expand_dims(pos_enc, 0)        # (1,H,W,D)
        pos_enc = tf.expand_dims(pos_enc, 1)        # (1,1,H,W,D)
        pos_enc = tf.tile(pos_enc, [b, t, 1, 1, 1]) # (B,T,H,W,D)
        x = x + pos_enc

        # 2) 通道注意力
        x_flat = tf.reshape(x, [-1, c])                       # (B*T*H*W, C)
        ch_att = self.channel_attention(x_flat)               # (B*T*H*W, C)
        ch_att = tf.reshape(ch_att, [b, t, h, w, c])          # (B,T,H,W,C)
        x_ch = x * ch_att

        # 3) 空间注意力
        sp_att = self.spatial_attention(tf.reshape(x_ch, [b*t, h, w, c]))  # (B*T,H,W,1)
        sp_att = tf.reshape(sp_att, [b, t, h, w, 1])          # (B,T,H,W,1)
        out = x_ch * sp_att

        # 4) 残差
        out = out + inputs

        if return_attention:
            # 通道注意力：压缩空间维度得到 [B,T,1,1,C]
            ch_vec = tf.reduce_mean(ch_att, axis=[2, 3], keepdims=True)
            return out, sp_att, ch_vec
        else:
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
    def __init__(self, return_attention=False):
        super(sstModel, self).__init__()
        self.return_attention = return_attention

        self.input_layer = Input(shape=(10, 40, 93, 73, 1))
        self.inception = InceptionBlk(12)
        self.att = GAMAttention(120)
        # 在这里为ConvGRU2D层添加L2正则化
        self.ConvLSTM2D0 = ConvGRU2D(filters=24, kernel_size=3, strides=1, padding='same',
                                     return_sequences=True)
        self.ConvLSTM2D = ConvGRU2D(filters=12, kernel_size=3, strides=1, padding='same',
                                    return_sequences=False)
        self.conv2 = Conv2D(filters=1, kernel_size=1, strides=1, padding='same')

    def call(self, x, training=None):
        x = self.ConvLSTM2D0(x, training=training)

        x1 = self.inception(x)
        x1_1 = tf.concat([x, x1], axis=-1)

        if self.return_attention:
            x2, spa, cha = self.att(x1_1, return_attention=True)
        else:
            x2 = self.att(x1_1, return_attention=False)

        x3 = self.ConvLSTM2D(x2, training=training)
        y  = self.conv2(x3)

        if self.return_attention:
            return y, spa, cha
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


import numpy as np
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf


# 定制化指标函数
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


# 定义预测下一天函数
def predict_next_day(model, x_input):
    predicted_sst = model.predict(np.expand_dims(x_input, axis=0))
    return np.squeeze(predicted_sst, axis=0)


import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input



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
        model = sstModel(return_attention=True)   # 打开注意力输出

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
    from datetime import datetime
    import os
    import numpy as np
    import tensorflow as tf
    import matplotlib

    matplotlib.use('Agg')  # 不弹窗、纯后台保存
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.mpl.gridliner as gridliner
    from matplotlib.colors import BoundaryNorm, ListedColormap

    from scipy.stats import pearsonr

    save_root = "./attention_maps_month"  # 可自行修改
    os.makedirs(save_root, exist_ok=True)

    num_days_to_predict = 12

    for i in range(x_test.shape[0] - num_days_to_predict):
        print(f"当前索引 i: {i}")
        x_test_sample = x_test[i:i + 1]  # shape: (1, 24, 92, 72, 1) -> numpy.ndarray

        # 模型推理
        y_pred, spa_att, cha_att = model(x_test_sample, training=False)

        sample_dir = os.path.join(save_root, f"sample_{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)


        att_24_mean = np.mean(spa_att[0].numpy(), axis=0)  # (92, 72, 1)
        att_24_mean = np.squeeze(att_24_mean)  # (92, 72)

        # -------------------------------
        # 使用 SST 均值创建陆地掩膜
        # -------------------------------
        sst_24_mean = np.mean(x_test_sample[0, :, :, :, 0], axis=0)  # (92, 72)
        land_mask = (sst_24_mean == 0)
        att_24_mean = np.ma.masked_where(land_mask, att_24_mean)

        # -------------------------------
        # 只绘制平均注意力图（单图）
        # -------------------------------
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(6, 5))  # 调整大小以适应单图
        ax = plt.axes(projection=proj)

        im = ax.imshow(att_24_mean, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                       transform=proj, cmap='RdYlBu_r', vmin=0, vmax=1,
                       interpolation='bilinear', origin='lower')     # cividis  viridis  plasma  turbo

        # 添加地理特征
        #ax.add_feature(cfeature.LAND, facecolor='lightgray')
        #ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=proj)
        ax.set_title('Mean Spatial Attention (12 months)', fontsize=14, pad=20)

        # 网格线
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = gridliner.LongitudeFormatter()
        gl.yformatter = gridliner.LatitudeFormatter()

        # 颜色条
        cbar = plt.colorbar(im, shrink=0.7, pad=0.08)
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        plt.tight_layout()
        fig.savefig(os.path.join(sample_dir, "avg_attention_only_month.png"), dpi=600, bbox_inches='tight')
        plt.close(fig)

    print("所有平均注意力图已保存到:", os.path.abspath(save_root))