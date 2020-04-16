import numpy as np
import tensorflow as tf
from argument_types import argPow2, activation
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
                      "args": ["argPow2", "avgFilterSize"]
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
                            "args": ['argPoolHeight', 'argPoolWidth'],
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
                            "args": ['argPoolHeight', 'argPoolWidth'],
                            "outputs": tf.Tensor,
                            "name": 'fractional_avg_pool'}

def activation(input):
    return tf.keras.layers.ReLU()(input)

operDict[activation] = {"inputs": [tf.keras.layers],
                      "output": tf.keras.layers,
                      "args": []
                    }
def dropout_layer(input):
	return tf.keras.layers.Dropout(rate=0.2)(input)

operDict[dropout_layer] = {"inputs": [tf.keras.layers],
                            "output": tf.keras.layers,
                            "args": []
                    }

def flatten_layer(input):
    return tf.keras.layers.Flatten()(input)

operDict[flatten_layer] = {"inputs": [tf.keras.layers],
							"output": tf.keras.layers,
							"args":[]
					}
def relu_func(input):
    # ReLu Non-linear activation function
    return tf.keras.layers.ReLu()(input)

def sigmoid_func(input):
    return tf.keras.activations.sigmoid()(input)

def batch_normalization_func(input):
    # Batch Normalization
    return tf.layers.batch_normalization(training=True)(input)

operDict[batch_normalization_func] = {"inputs": [tf.keras.layers],
                                      "outputs": tf.keras.layers,
                                      "args": []
                    }

def global_avg_pool_layer(input):
    return avg_pool_layer(input)

def add_tensors(a, b):
    return tf.keras.layers.Add()([a, b])
operDict[add_tensors] = {"inputs": [tf.keras.layers, tf.keras.layers],
						"outputs": tf.keras.layers,
						"args": []
						}
def sub_fa2a(a, b):
    return np.subtract(a,b)

def sub_tensors(a,b):
    return tf.keras.layers.Subtract([a,b])
operDict[sub_tensors] = {"inputs": [tf.keras.layers, tf.keras.layers],
						"outputs": tf.keras.layers,
						"args": []
						}
def mult_tensors(a,b):
    return keras.layers.Multiply([a,b])
operDict[mult_tensors] = {"inputs": [tf.keras.layers, tf.keras.layers],
						"outputs": tf.keras.layers,
						"args": []
						}

def ceil_greyscale_norm(input):
    return input/255

operDict[ceil_greyscale_norm] = {"inputs": [np.ndarray],
                            "outputs": np.ndarray,
                            "args": []
                        }
def max_pool_layer(input):
    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    if input.shape[1].value == 1:
        return input_tensor
    return tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="valid")(input)
operDict[max_pool_layer] = {"inputs": [tf.keras.layers],
						"outputs": tf.keras.layers,
						"args": []
						}

def avg_pool_layer(input):
	return tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="valid")(input)
    #return tf.layers.average_pooling2d(inputs=input_tensor, pool_size=[2,2], strides=2)

operDict[avg_pool_layer] = {"inputs": [tf.keras.layers],
						"outputs": tf.keras.layers,
						"args": []
						}

def identity_layer(input):
    print('before identity: ', input)
    output_tensor = model.add(Lambda(lambda input: input))
    print('after identity: ', output_tensor)
    return output_tensor

operDict[identity_layer] = {"inputs": [tf.keras.layers],
						"outputs": tf.keras.layers,
						"args": []
						}



