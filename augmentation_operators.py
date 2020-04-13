import Augmentor
from Augmentor.Operations import Operation
import numpy as np

class Normalize(Operation):
    # Here you can accept as many custom parameters as required:
    def __init__(self):
        # Call the superclass's constructor (meaning you must
        # supply a probability value):
        Operation.__init__(self, probability = 1)

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
        Operation.__init__(self, probability = 1)
        self.resModel = resModel
        self.cutModel = None

    # Your class must implement the perform_operation method:
    def perform_operation(self, images):
        # Return the image so that it can further processed in the pipeline:
        images = np.asarray(images)
        if not self.resModel:
            # specify self.resModel
            self.resModel = ResNet152V2(include_top=False, weights='imagenet',input_tensor=None,
                                input_shape=images[0].shape)
        return self.resModel.predict(images)[:, 0, 0, :]