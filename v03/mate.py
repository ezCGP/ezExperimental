'''
don't include self in any of these classes. we don't plan on making instances of the class.
'''

# packages
import sys
import os
from abc import ABC, abstractmethod
from numpy import random as rnd
import numpy as np

# scripts
import mate_methods


class MateDefinition(ABC):
    '''
    words
    '''
    def __init__(self,
                prob_mate = 0.10):
        self.prob_mate = prob_mate

    @abstractmethod
    def mate(self, parent1, parent2, block_index: int):
        pass



class WholeMateOnly(MateDefinition):
    '''
    each pair of block/parents will mate w/prob 25%

    if they mate, they will only mate with whole_block()
    '''
    def __init__(self):
        prob_mate = 1.0
        MateDefinition.__init__(self, prob_mate)

    def mate(self, parent1, parent2, block_index: int):
        return mate_methods.whole_block(parent1, parent2, block_index)


class NoMate(MateDefinition):
    def __init__(self):
        prob_mate = 0
        MateDefinition.__init__(self, prob_mate)

    def mate(self, parent1, parent2, block_index: int):
        return []
