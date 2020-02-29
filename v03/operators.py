'''
OperatorDefinition:

Definition for the Operators interface, which collects primitives from various user-defined modules and creates operator-weight pairs that will be used to initialize genome.

Specific problems (e.g. symbolic regression) are implemented as an example on how to initiate an Operators set definition. 

'''

# packages
import os
import sys
from typing import List

# scripts
from utils import build_weights
import simple_numpy


class OperatorDefinition():
    '''
    An interface that outlines the data structure that stores operators for a given machine learning task (e.g. primitive neural network layers, functions)
    '''
    def __init__(self,
                modules: List):
        '''
        :param modules: the filenames from modules
        '''
        self.build_operDict(modules)
        self.build_weights()

    def build_operDict(self, modules: List):
        '''
        Import the operator dict from every module in the list and return

        Creates a single operator dictionary from all of the individual modules
        '''
        self.operator_dict = {}
        for file_name in modules:
            module = __import__(file_name)
            self.operator_dict.update(moduel.operDict)
            del module 

    def build_weights(self):
        '''
        Converts weights into normalized relative weights and reassigns them to self.operator_dict
        '''
        weights = []
        for operator in self.operator_dict:
            weights.append(operator['weight'])

        # cast to numpy to normalize
        weights = np.array(weights) 

        # normalize the weights
        weights /= np.sum(weights)

        # reassign the weights to operators dictionary
        for operator, weight in self.operator_dict, weights:
           operator['weight'] = weight 

class SymbRegressionNoArgs(OperatorDefinition):
    '''
    Operator definition for the Symbolic Regression problem
    '''
    def __init__(self):
        # example module user defines for symbolic regression operators
        modules = ['symbolic_regression_operators'] 
        OperatorDefinition.__init__(self,
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
