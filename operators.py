'''
words
'''

# packages
import os
import sys
from typing import List

# scripts
from utils import build_weights
import simple_numpy
#import tensorflow_operator
#somehow importing^ before pipeline_operators made a difference with tf.keras.layers.__name__
#so moved the import after
import pipeline_operators
import tensorflow_operator

class OperatorDefinition():
    '''
    words
    '''
    def __init__(self,
                operators: List,
                weights: List,
                modules: List):
        self.build_operDict(modules)
        self.operators = operators
        self.weights = weights


    def build_operDict(self, modules: List):
        '''
        import the operator dict from every module in the list and return
        '''
        self.operator_dict = {}
        for oper_py in modules:
            _ = __import__(oper_py)
            self.operator_dict.update(_.operDict)
            del _


class SymbRegressionNoArgs(OperatorDefinition):
    '''
    words
    '''

    def __init__(self):
        modules = ['simple_numpy']
        weight_dict = {simple_numpy.add_ff2f: 1,
                       simple_numpy.add_fa2a: 1,
                       simple_numpy.add_aa2a: 1,
                       simple_numpy.sub_ff2f: 1,
                       simple_numpy.sub_fa2a: 1,
                       simple_numpy.sub_aa2a: 1,
                       simple_numpy.mul_ff2f: 1,
                       simple_numpy.mul_fa2a: 1,
                       simple_numpy.mul_aa2a: 1}
        operators, weights = build_weights(weight_dict)
        OperatorDefinition.__init__(self,
                                    operators,
                                    weights,
                                    modules)


class TFOps(OperatorDefinition):
    '''
    Experimental Tensorflow Operator Defition
    This class is used by problem_interface in order to run
    an evolution. Specifies tensorflow primitives.
    '''

    def __init__(self):
        modules = ['tensorflow_operator']
        weight_dict = {tensorflow_operator.dense_layer: 1,
                       tensorflow_operator.conv_layer: 1,
                       tensorflow_operator.fractional_max_pool: 1,

                        } #  TODO fix this. See issue

        operators, weights = build_weights(weight_dict)
        OperatorDefinition.__init__(self,
                                    operators,
                                    weights,
                                    modules)


class AugmentationOps(OperatorDefinition):

    def __init__(self):
        modules = ['pipeline_operators']
        weight_dict = {pipeline_operators.rotate: 1,
                       pipeline_operators.horizontal_flip: 1,
                       pipeline_operators.random_crop: 1}

        operators, weights = build_weights(weight_dict)
        OperatorDefinition.__init__(self,
                                    operators,
                                    weights,
                                    modules)


class PreprocessingOps(OperatorDefinition):

    def __init__(self):
        modules = ['pipeline_operators']
        weight_dict = {pipeline_operators.normalize: 1,
                       pipeline_operators.res_net_norm:1,
                       pipeline_operators.res_net: 1}

        operators, weights = build_weights(weight_dict)
        OperatorDefinition.__init__(self,
                                    operators,
                                    weights,
                                    modules)


'''
class SymbRegressionWithArgs(OperatorDefinition):
    #words
    def __init__(self):
        modules = ['simple_numpy']
        weight_dict = {simple_numpy.add_aa2a: 1,
                    simple_numpy.sub_ff2f: 1,
                    simple_numpy.sub_fa2a: 1,
                    simple_numpy.sub_aa2a: 1,
                    simple_numpy.mul_ff2f: 1,
                    simple_numpy.mul_fa2a: 1,
                    simple_numpy.mul_aa2a: 1}
        operators, weights = build_weights(weight_dict)
        OperatorDefinition.__init__(self,
                                    operators,
                                    weights,
                                    modules)
'''