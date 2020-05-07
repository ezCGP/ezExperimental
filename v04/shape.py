'''
this will be the only 'Definition' type that will be instantiated because it is 'attribute' focused and not 'method' focused
'''

# packages
import os
import sys
import numpy as np

# scripts

class ShapeMetaDefinition():
    '''
    a lot of this is just to help fill attributes of a block
    like number of nodes, acceptable input/output datatypes, etc
    '''
    def __init__(self,
                input_dtypes: list=[],
                output_dtypes: list=[],
                main_count: int=20):
        self.input_dtypes = input_dtypes
        self.input_count = len(input_dtypes)
        self.output_dtypes = output_dtypes
        self.output_count = len(output_dtypes)
        self.main_count = main_count
        self.genome_count = self.input_count+self.output_count+self.main_count


'''
shape_A = ShapeDefinition([np.float64, np.float64],
                        [np.float64],
                        25)'''
class ShapeA(ShapeMetaDefinition):
    def __init__(self):
        input_dtypes = [np.float64, np.ndarray]
        output_dtypes = [np.ndarray]
        main_count = 25
        ShapeMetaDefinition.__init__(self,
                                input_dtypes,
                                output_dtypes,
                                main_count)