import Augmentor
from Augmentor.Operations import Operation
import numpy as np
from argument_types import argFloat
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.resnet50 import preprocess_input

operDict = {}

class Normalize(Operation):
    # Here you can accept as many custom parameters as required:
    def __init__(self):
        # Call the superclass's constructor (meaning you must
        # supply a probability value):
        Operation.__init__(self, probability=1)

    # Your class must implement the perform_operation method:
    def perform_operation(self, images):
        # Return the image so that it can further processed in the pipeline:
        augmentedImages = []
        for image in images:
            modImage = np.asarray(image) / 255.0
            augmentedImages.append(modImage)
        return augmentedImages


class ResNet(Operation):
    """
    Purpose of function is to add an Augmentor primtive that is literally the output of a fully trained neural network
    """

    # Here you can accept as many custom parameters as required:
    def __init__(self, resModel=None):
        # Call the superclass's constructor (meaning you must
        # supply a probability value):
        Operation.__init__(self, probability=1)
        self.resModel = resModel
        self.cutModel = None

    # Your class must implement the perform_operation method:
    def perform_operation(self, images):
        # Return the image so that it can further processed in the pipeline:
        images = np.asarray(images)
        resModel = ResNet152V2(include_top=False, weights='imagenet', input_tensor=None,
                                    input_shape=images[0].shape)
        return resModel.predict(images)[:, 0, 0, :]

class ResNetNorm(Operation):
    """
    Purpose of function is to add an Augmentor primtive that is literally the output of a fully trained neural network
    """

    # Here you can accept as many custom parameters as required:
    def __init__(self):
        # Call the superclass's constructor (meaning you must
        # supply a probability value):
        Operation.__init__(self, probability=1)

    # Your class must implement the perform_operation method:
    def perform_operation(self, images):
        # Return the image so that it can further processed in the pipeline:
        images = np.asarray(images)
        return preprocess_input(images)

"""Augmentation Primtives"""

def rotate(p, probability=.5, max_left_rotation=10, max_right_rotation=10):
    p.rotate(probability, max_left_rotation, max_right_rotation)
    return p


operDict[rotate] = {"inputs": [Augmentor.Pipeline],
                      "output": Augmentor.Pipeline,
                      "args": [argFloat]
                    }

"""https://arxiv.org/pdf/1912.11370v2.pdf"""
def horizontal_flip(p, probability = .5):
    p.flip_left_right(probability)
    return p

operDict[horizontal_flip] = {"inputs": [Augmentor.Pipeline],
                      "output": Augmentor.Pipeline,
                      "args": [argFloat]
                    }

"""https://arxiv.org/pdf/1912.11370v2.pdf"""
def random_crop(p, probability=.2):
    p.flip_left_right(probability)
    return p

operDict[random_crop] = {"inputs": [Augmentor.Pipeline],
                    "output": Augmentor.Pipeline,
                    "args": [argFloat]  # this will choose values between 0 and 1.
                                        # This may not be what we want though as 1 would black out the entire image
                    }


"""Preprocessing primitives"""
def normalize(p: Augmentor.Pipeline, probability=.5):
    p.add_operation(Normalize())
    return p
operDict[normalize] = {"inputs": [Augmentor.Pipeline],
                    "output": Augmentor.Pipeline,
                    "args": []  # this will choose values between 0 and 1.
                                        # This may not be what we want though as 1 would black out the entire image
                    }

def res_net_norm(p):
    p.add_operation(ResNetNorm())
    return p

operDict[res_net_norm] = {"inputs": [Augmentor.Pipeline],
                       "output": Augmentor.Pipeline,
                       "args": []
                       }
"""https://keras.io/applications/"""
def res_net(p):
    p.add_operation(ResNet())
    return p

operDict[res_net] = {"inputs": [Augmentor.Pipeline],
                       "output": Augmentor.Pipeline,
                       "args": []
                       }

