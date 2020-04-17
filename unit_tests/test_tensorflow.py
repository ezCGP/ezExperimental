import unittest
import tensorflow_operator
import tensorflow as tf
import numpy as np
class TestStringMethods(unittest.TestCase):

    def test_input_layer(self):
        try:
            input_tensor = tf.keras.layers.Input([3, 32, 32])
            inputLayer_out = tensorflow_operator.input_layer(input_tensor)
            model = tf.keras.Model(input_tensor, inputLayer_out, name="dummy")
            fake_input = np.zeros((10000, 3, 32, 32), dtype =np.float64)
            out = model.predict(fake_input)
            assert(1 == 1)
        except Exception as e:
            assert(1 == 0)

    def test_dense(self):
        try:
            input_tensor = tf.keras.layers.Input([3, 32, 32])
            dense_out = tensorflow_operator.dense_layer(input_tensor)
            model = tf.keras.Model(input_tensor, dense_out, name="dummy")
            fake_input = np.zeros((10000, 3, 32, 32), dtype =np.float64)
            out = model.predict(fake_input)
            assert(1 == 1)
        except Exception as e:
            assert(1 == 0)

    def test_conv_layer(self):
        try:
            input_tensor = tf.keras.layers.Input([3, 32, 32])
            conv_layer_out = tensorflow_operator.conv_layer(input_tensor)
            model = tf.keras.Model(input_tensor, conv_layer_out, name="dummy")
            fake_input = np.zeros((10000, 3, 32, 32), dtype =np.float64)
            out = model.predict(fake_input)
            assert(1 == 1)
        except Exception as e:
            assert(1 == 0)

    def test_fractional_max_pool(self):
        try:
            input_tensor = tf.keras.layers.Input([3, 32, 32])
            frac_max_pool_out = tensorflow_operator.fractional_max_pool(input_tensor)
            model = tf.keras.Model(input_tensor, frac_max_pool_out, name="dummy")
            fake_input = np.zeros((10000, 3, 32, 32), dtype =np.float64)
            out = model.predict(fake_input)
            assert(1 == 1)
        except Exception as e:
            assert(1 == 0)

    def test_fractional_avg_pool(self):
        try:
            input_tensor = tf.keras.layers.Input([3, 32, 32])
            frac_avg_pool_out = tensorflow_operator.fractional_avg_pool(input_tensor)
            model = tf.keras.Model(input_tensor, frac_avg_pool_out, name="dummy")
            fake_input = np.zeros((10000, 3, 32, 32), dtype =np.float64)
            out = model.predict(fake_input)
            assert(1 == 1)
        except Exception as e:
            assert(1 == 0)

if __name__ == '__main__':
    unittest.main()