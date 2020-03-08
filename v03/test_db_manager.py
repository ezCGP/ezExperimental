import numpy as np
import Augmentor
from Augmentor.Operations import Operation

from database.db_manager import DbManager
from database.db_config import DbConfig


db_config = DbConfig()
manager = DbManager(db_config)

# train, test, val = manager.load_CIFAR10()
# x_train, y_train = train
# x_test, y_test = test
# x_val, y_val = val
#
# x_train, y_train = np.array([x_train]), np.array(y_train)  # this is such a weird shape requirement
# x_test, y_test = np.array([x_test]), np.array([y_test])
# x_val, y_val = np.array(x_val), np.array(y_val)
# x_train = x_train[0].astype(np.uint8)


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


dataset = manager.load_CIFAR10()

dataset.train_pipeline.rotate90(.1)
dataset.train_pipeline.add_operation(Normalize())
dataset.test_pipeline.add_operation(Normalize())

x_batch, y_batch = dataset.next_batch_train(128)  # training data is randomly rotated and normalized
print("X_batch shape", x_batch.shape)
x_val_norm, y_val = dataset.preprocess_test_data()  # testing data is just normalized
print("X val norm shape", x_val_norm.shape)
dataset.clear_data()