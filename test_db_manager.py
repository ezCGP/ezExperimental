import numpy as np
import Augmentor
from Augmentor.Operations import Operation

from database.ezDataLoader import load_CIFAR10




# Create your new operation by inheriting from the Operation superclass:
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

dataset = load_CIFAR10(.8, .2) # .8 training percent, .2 validation
dataset.train_pipeline.rotate90(.1)
dataset.train_pipeline.add_operation(Normalize())
dataset.test_pipeline.add_operation(Normalize())

x_batch, y_batch = dataset.next_batch_train(128)  # training data is randomly rotated and normalized
print("X_batch shape", x_batch.shape)
x_val_norm, y_val = dataset.preprocess_test_data()  # testing data is just normalized
print("X val norm shape", x_val_norm.shape)
dataset.clear_data()