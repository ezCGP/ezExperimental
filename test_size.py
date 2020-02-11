import numpy as np
from numpy import random as rnd
import sys
from pympler import asizeof, classtracker
from abc import ABC, abstractmethod


num_blocks = 3
num_main_nodes = 20
num_input_nodes = 3
num_output_nodes = 1
num_nodes = num_main_nodes + num_input_nodes + num_output_nodes
num_args = 50


primitives = [np.add, np.subtract]
args = [np.float64, np.int64]


class Block_Material():
    def __init__(self):
        self.genome = [None]*num_nodes
        for i in range((-1*num_input_nodes),0):
            self.genome[i] = "InputPlaceholder"
        for i in range(num_main_nodes):
            node_dict = {'function': rnd.choice(primitives),
                        'inputs': list(rnd.choice(range(-1*num_input_nodes, i), size=2)),
                        'args': list(rnd.choice(range(num_args), size=2))}
            self.genome[i] = node_dict
        for i in range(num_main_nodes, num_main_nodes+num_output_nodes):
            self.genome[i] = rnd.choice(range(num_main_nodes))

        self.args = list(rnd.choice(args, size=num_args))

        self.need_evaluate = True


    def __setitem__(self, node_index, value):
        self.genome[node_index] = value

    def __getitem__(self, node_index):
        return self.genome[node_index]


class Individual_Material():
    def __init__(self):
        self.blocks = []
        for block_def in range(num_blocks):
            self.blocks.append(Block_Material())

        self.fitness = self.Fitness()

    def __getitem__(self, block_index):
        return self.blocks[block_index]


    def need_evaluate(self):
        for i in range(num_blocks):
            if self[i].need_evaluate:
                return True
        return False


    class Fitness(object):
        '''
        the NSGA taken from deap requires a Fitness class to hold the values.
        so this attempts to recreate the bare minimums of that so that NSGA
        or (hopefully) any other deap mutli obj ftn handles this Individual class
        http://deap.readthedocs.io/en/master/api/base.html#fitness
        '''

        def __init__(self):
            self.values = () #empty tuple

        # check dominates
        def dominates(self, other):
            a = np.array(self.values)
            b = np.array(other.values)
            # 'self' must be at least as good as 'other' for all objective fnts (np.all(a>=b))
            # and strictly better in at least one (np.any(a>b))
            return np.any(a < b) and np.all(a <= b)



ting = Individual_Material()
print("sys", sys.getsizeof(ting))
print("pympler", asizeof.asizeof(ting))
print(asizeof.asized(ting, detail=1).format())



class Ting():
    def __init__(self,
                aa: int = 97,
                bb: int = 98,
                cc: int = 99):
        self.a = aa
        self.b = bb
        self.c = cc

    def randommethod(self):
        pass

class Other():
    def __init__(self, ting: Ting):
        print(ting.__dict__)
        for name, val in ting.__dict__.items():
            self.__dict__[name] = val

ori = Ting(1.1,2,3)
oth = Other(ori)


class ThisDef(ABC):
    prob = .3

    @abstractmethod
    def hi():
        pass

class WithThis(ThisDef):
    prob = 400
    print("yo")
    # pd exists part of this class but not externally
    # BUT it does get run even if you don't call WithThis
    import pandas as pd
    df = pd.DataFrame()
    def hi():
        print("hi")

    def helloplus():
        print("hello")
        import scipy as sci
        WithThis.hi()