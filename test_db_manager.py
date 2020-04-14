
import numpy as np
from Augmentor.Operations import Operation

from PIL import Image
import time
from database.ezDataLoader import load_CIFAR10

from tensorflow.keras.applications.resnet_v2 import ResNet152V2


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

# Create your new operation by inheriting from the Operation superclass:
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
        return self.resModel.predict(images)


dataset = load_CIFAR10(.95, .05) # .8 training percent, .2 validation

#  Data Augmentation
dataset.augmentation_pipeline.rotate90(.99)  # rotate 99% of images
dataset.augmentation_pipeline.skew(.5)  # shew 50% of images


#  Preprocessing Step
dataset.preprocess_pipeline.add_operation(Normalize())
dataset.preprocess_pipeline.add_operation(ResNet())  # trained neural network can be thought of as advanced feature extraction

start = time.time()
x_batch, y_batch = dataset.next_batch_train(128)  # training data is randomly rotated and skewed. Also uses preprocess_pipeline
# img = Image.fromarray(x_batch[0], "RGB") # can only view if not gone through ResNet
# img.show()
print("X_batch shape", x_batch.shape)
x_val_norm, y_val = dataset.preprocess_test_data()  # testing data is just normalized and passed through ResNet
print("X val norm shape", x_val_norm.shape)
dataset.clear_data()
print("Time elapsed", time.time() - start)






