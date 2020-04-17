import numpy as np
import tensorflow as tf
from argument_types import argPow2, activation, argFilterSize, argPoolHeight, argPoolWidth
operDict = {}

def input_layer(input_tensor):
    return tf.keras.layers.InputLayer()(input_tensor)

operDict[input_layer] = {"inputs": [tf.keras.layers],
                      "output": tf.keras.layers,
                      "args": []
                    }

def dense_layer(input_tensor, num_units=128, activation="relu"):
    # Flatten tensor into a batch of vectors
    pool2_flat = tf.keras.layers.Flatten()(input_tensor)
    # Densely connected layer with 1024 neurons
    logits = tf.keras.layers.Dense(units=num_units, activation=activation)(pool2_flat)
    return logits

operDict[dense_layer] = {"inputs": [tf.keras.layers],
                      "output": tf.keras.layers,
                      "args": [argPow2, activation]
                    }


def conv_layer(input_tensor, filters=64, kernel_size=3, activation = tf.nn.relu):
    kernel_size = (kernel_size, kernel_size)
    # Convolutional Layer
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding="same",
                            activation=activation, data_format="channels_last")(input_tensor)

operDict[conv_layer] = {"inputs": [tf.keras.layers],
                      "output": tf.keras.layers,
                      "args": [argPow2, argFilterSize, activation]
                    }

#Functions - need max_pool_layer
# def concat_func(data1, data2):
#     # Concatenates two feature maps in the channel dimension
#     # If one feature map is larger, we downsample it using max pooling
#     if data1.shape[1].value > data2.shape[1].value:
#         data1 = max_pool_layer(data1)
#         return concat_func(data1, data2)
#     elif data1.shape[1].value < data2.shape[1].value:
#         data2 = max_pool_layer(data2)
#         return concat_func(data1, data2)
#     else:
#         return tf.concat([data1, data2], 3)
#
# operDict[concat_func] = {"inputs": [tf.keras.layers, tf.keras.layers],
#                             "args": [],
#                             "outputs": tf.keras.layers,
#                             "name": 'concatFunc',
#                             "include_labels": False}

# TODO: see error PACE, start here
def fractional_max_pool(input_tensor, pool_height = 2.0, pool_width = 2.0):
    if input_tensor.shape[1].value == 1:
        return input_tensor
    pooling_ratio = [1.0, pool_height, pool_width, 1.0]      # see args.py for mutation limits
    pseudo_random = True        # true random underfits when combined with data augmentation and/or dropout
    overlapping = True          # overlapping pooling regions work better, according to 2015 Ben Graham paper
    # returns a tuple of Tensor objects (output, row_pooling_sequence, col_pooling_sequence
    return tf.nn.fractional_max_pool(input_tensor, pooling_ratio, pseudo_random, overlapping)[0]

operDict[fractional_max_pool] = {"inputs": [tf.Tensor],
                            "args": [argPoolHeight, argPoolWidth],
                            "outputs": tf.Tensor,
                            "name": 'fractional_max_pool'}

def fractional_avg_pool(input_tensor, pool_height = 2.0, pool_width = 2.0):
    if input_tensor.shape[1].value == 1:
        return input_tensor
    pooling_ratio = [1.0, pool_height, pool_width, 1.0]
    pseudo_random = True
    overlapping = True
    # returns a tuple of Tensor objects (output, row_pooling_sequence, col_pooling_sequence)
    return tf.nn.fractional_avg_pool(input_tensor, pooling_ratio, pseudo_random, overlapping)[0]

operDict[fractional_avg_pool] = {"inputs": [tf.Tensor],
                            "args": [argPoolHeight, argPoolWidth],
                            "outputs": tf.Tensor,
                            "name": 'fractional_avg_pool'}

def activation(input):
    return tf.keras.layers.ReLU()(input)

operDict[activation] = {"inputs": [tf.keras.layers],
                      "output": tf.keras.layers,
                      "args": []
                    }

