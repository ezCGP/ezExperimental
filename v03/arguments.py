'''
Where we specify which datatype to include
'''

# packages
from typing import List

# scripts
import argument_types
from utils import build_weights




class ArgumentDefinition():
    '''
    words
    '''
    def __init__(self,
                arg_count: int,
                arg_types: List,
                arg_weights: List):
        self.arg_count = arg_count
        self.arg_alltypes = arg_types
        self.arg_weights = arg_weights
        self.fill_args()


    def get_arg_weights(weight_dict):
        '''
        only works before __init__ called
        '''
        args, weights = build_weights(weight_dict)


    def fill_args(self):
        '''
        note it only fills it by the data type class not instances of the argtype
        '''
        start_point = 0
        end_point = 0
        self.arg_types = [None]*self.arg_count
        for arg_type, arg_weights in zip(self.arg_alltypes, self.arg_weights):
            end_point += int(arg_weight*self.arg_count)
            for arg_index in range(start_point, end_point):
                self.arg_types[arg_index] = arg_type
            start_point = end_point
        if end_point != self.arg_count:
            # prob some rounding errors then
            sorted_byweight = np.argsort(self.arg_weights)[::-1] # sort then reverse
            for i, arg_index in enumerate(range(end_point, self.arg_count)):
                arg_class = self.arg_alltypes[sorted_byweight[i]]
                self.arg_types[arg_indx] = arg_class
        else:
            pass



class Size50(ArgumentDefinition):
    def __init__(self):
        arg_count = 50
        arg_dict = {argument_types.argInt: 1,
                    argument_types.argPow2: 1}
        args, weights = ArgumentDefinition.get_arg_weights(arg_dict)
        ArgumentDefinition.__init__(self,
                                    arg_count,
                                    args,
                                    weights)


class NoArgs(ArgumentDefinition):
    def __init__(self):
        arg_count = 0 
        ArgumentDefinition.__init__(self,
                                    arg_count,
                                    [],
                                    [])