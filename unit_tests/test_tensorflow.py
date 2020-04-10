import unittest
import tensorflow_operator
import tensorflow as tf
import numpy as np
class TestStringMethods(unittest.TestCase):

    def test_dense(self):
        try:
            input_ = tf.keras.layers.Input([3, 32, 32])
            dense_out = tensorflow_operator.dense_layer(input_)
            model = tf.keras.Model(input_, dense_out, name="dummy")
            fake_input = np.zeros((10000, 3, 32, 32), dtype =np.float64)
            out = model.predict(fake_input)
            assert(1 == 1)
        except Exception as e:
            assert(1 == 0)

if __name__ == '__main__':
    unittest.main()