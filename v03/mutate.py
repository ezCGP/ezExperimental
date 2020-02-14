'''
words
'''

# packages
import os
import sys
from abc import ABC, abstractmethod
from numpy import random as rnd

# scripts
import mutate_methods


class MutateDefinition(ABC):
    '''
    made into an abstract class

    anything with @abstractmethod will have to be filled in by
    whatever class that inherits from MutateDefinition
    '''
    def __init__(self,
                prob_mutate = 0.30,
                num_mutants = 4):
        self.prob_mutate = prob_mutate
        self.num_mutants = num_mutants

    @abstractmethod
    def mutate(self, indiv, block_index: int):
        pass


class MutateA(MutateDefinition):
    '''
    each block has a prob of mutate at 10%

    then if we choose to mutate that block,
    it will use muatate_1 25% of the time and
    mutate_2 75% of the time
    '''
    def __init__(self):
        prob_mutate = 0.10
        MutateDefinition.__init__(self, prob_mutate)


    def mutate(self, indiv, block_index: int):
        roll = rnd.random()
        if roll < 0.25:
            mutate_methods.mutate_1(indiv, block_index)
        else:
            mutate_methods.mutate_2(indiv, block_index)


class MutateB(MutateDefinition):

    def __init__(self):
        prob_mutate = 1.0
        MutateDefinition.__init__(self, prob_mutate)


    def mutate(self, indiv, block_index: int, block_def):
        roll = rnd.random()
        if roll < (1/4):
            print("SINGLE INPUT")
            mutate_methods.mutate_single_input(indiv, block_index, block_def)
        elif roll < (2/4):
            print("SINGLE ARG VALUE")
            mutate_methods.mutate_single_argvalue(indiv, block_index, block_def)
        elif roll < (3/4):
            print("SINGLE ARG INDEX")
            mutate_methods.mutate_single_argindex(indiv, block_index, block_def)
        else:
            print("SINGLE FTN")
            mutate_methods.mutate_single_ftn(indiv, block_index, block_def)