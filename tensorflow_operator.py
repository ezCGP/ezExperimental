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



