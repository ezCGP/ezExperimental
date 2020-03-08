import numpy as np
import tensorflow as tf
operDict = {}

def dense_layer(input):
    return tf.keras.layers.Dense(100)(input)
operDict[dense_layer] = {"inputs": [tf.keras.layers],
                      "output": tf.keras.layers,
                      "args": []
                    }
def activation(input):
    return tf.keras.layers.ReLU()(input)
operDict[activation] = {"inputs": [tf.keras.layers],
                      "output": tf.keras.layers,
                      "args": []
                    }