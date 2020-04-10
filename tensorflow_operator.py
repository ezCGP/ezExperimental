import numpy as np
import tensorflow as tf
from argument_types import argPow2 #
operDict = {}

def dense_layer(input):
    return tf.keras.layers.Dense(100)(input) # hardcoded to 100
operDict[dense_layer] = {"inputs": [tf.keras.layers],
                      "output": tf.keras.layers,
                      "args": []
                    }

# def dense_layer(input_tensor):
#     num_units = 128
#     activation = "relu"
#     # Flatten tensor into a batch of vectors
#     pool2_flat = layers.Flatten(input_tensor)
#     # Densely connected layer with 1024 neurons
#     logits = layers.dense(inputs=pool2_flat, units=num_units, activation = activation)
#     return logits
#
# operDict[dense_layer] = {"inputs": [tf.keras.layers],
#                       "output": tf.keras.layers,
#                       "args": [argPow2]
#                     }

def activation(input):
    return tf.keras.layers.ReLU()(input)
operDict[activation] = {"inputs": [tf.keras.layers],
                      "output": tf.keras.layers,
                      "args": []
                    }

