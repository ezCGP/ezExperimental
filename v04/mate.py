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
sys.path.insert(1, "./utilities_gp")
import mate_methods


class MateDefinition(ABC):
    '''
    REQUIREMENTS/EXPECTATIONS

    Individual Mate class:
     * if a block is mated, need_evaluate should be set to True at this level no matter what
     * there is a wide variation of ways we can mate so deepcopies should occur at the mate_methods level, not here or block
     * inputs: instance of IndividualDefinition and then two instances of IndividualMaterial as the parents 
     * returns: a list of new offspring individuals or an empty list

    Block Mate class:
     * in __init__ will assign a prob_mate attribute for that block
     * as above, we should not deepcopy at all here; we assume that the mate_method itself will handle that and simply return the list
        output by the select mate_method
     * inputs: the 2 parents as instances of IndividualMaterial, integer for the i^th block we want to mate
     * returns: a list of offspring output by the selected mate_method
    '''
    def __init__(self):
        pass

    @abstractmethod
    def mate(self, parent1, parent2, block_index: int):
        pass


class IndividualMateA(MateDefinition):
    '''
    words
    '''
    def __init__(self):
        pass

    def mate(self, indiv_def, parent1, parent2):
        all_children = []
        for block_index in range(indiv_def.block_count):
            roll = rnd.random()
            if roll < indiv_def[block_index].mate_def.prob_mate:
                #children: List() = indiv_def[block_index].mate_def.mate(parent1, parent2, block_index)
                children = indiv_def[block_index].mate_def.mate(parent1, parent2, block_index)
                # for each child, we need to set need_evaluate on all nodes from the mated block and on
                for child in children:
                    for block_i in range(block_index, indiv_def.block_count):
                        child[block_i].need_evaluate = True
                all_children += children #join the lists
        return all_children


class BlockWholeMateOnly(MateDefinition):
    '''
    each pair of block/parents will mate w/prob 25%

    if they mate, they will only mate with whole_block()
    '''
    def __init__(self):
        self.prob_mate = 1.0

    def mate(self, parent1, parent2, block_index: int, block_def):
        return mate_methods.whole_block(parent1, parent2, block_index, block_def)


class BlockNoMate(MateDefinition):
    def __init__(self):
        self.prob_mate = 0

    def mate(self, parent1, parent2, block_index: int, block_def):
        return []
