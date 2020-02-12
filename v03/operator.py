'''
words
'''

# packages
import os
import sys

# scripts
from utils import build_weights
import simple_numpy



class OperatorDefinition():
    '''
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
        self.operDict = {}
        for oper_py in modultes:
            _ = __import__(oper_py)
            self.operDict.update(_.operDict)
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

'''
class SymbRegressionWithArgs(OperatorDefinition):
    '''
    #words
    '''
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