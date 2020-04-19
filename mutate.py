'''
words
'''

# packages
import os
import sys
from copy import deepcopy
from abc import ABC, abstractmethod
from numpy import random as rnd

# scripts
sys.path.insert(1, "./utilities_gp")
import mutate_methods


class MutateDefinition(ABC):
    '''
    REQUIREMENTS/EXPECTATIONS

    Individual Mutate class:
     * deepcopies should always happen at this level and the copied individual sent to the blocks to be mutated in-place
     * RE setting need_evaluate to True after mutation, this is expected to occur at the mutate_method level because it is possible for some mutations
        to not mutate active nodes so need_evaluate could remain False
     * inputs: instance of IndividualDefinition and instance of IndividualMaterial
     * returns: a list of new mutated individuals or an empty list

    Block Mutate class:
     * in __init__ will assign a prob_mutate and num_mutants attribute for that block
     * this method will mutate the given individual in-place. do not deepcopy here
     * inputs: instance of IndividualMaterial, integer for the i^th block we want to mutate
     * returns: nothing as the mutation should occur in-place to the given individual
    '''
    def __init__(self):
        pass

    @abstractmethod
    def mutate(self):
        pass


class InidividualMutateA(MutateDefinition):
    def __init__(self):
        pass

    def mutate(self, indiv_def, indiv):
        mutants = []
        for block_index in range(indiv_def.block_count):
            roll = rnd.random()
            if roll < indiv_def[block_index].mutate_def.prob_mutate:
                for _ in range(indiv_def[block_index].num_mutants):
                    mutant = deepcopy(indiv)
                    indiv_def[block_index].mutate(mutant, block_index)
                    mutants.append(mutant)
        return mutants



class BlockMutateA(MutateDefinition):

    def __init__(self):
        self.prob_mutate = 1.0
        self.num_mutants = 1  # to reduce the number of mutants

    def mutate(self, indiv, block_index: int, block_def):
        roll = rnd.random()
        if roll < (1/2):
            #print("SINGLE INPUT")
            mutate_methods.mutate_single_input(indiv, block_index, block_def)
        else:
            #print("SINGLE FTN")
            mutate_methods.mutate_single_ftn(indiv, block_index, block_def)


class BlockMutateB(MutateDefinition):

    def __init__(self):
        self.prob_mutate = 1.0
        self.num_mutants = 4


    def mutate(self, indiv, block_index: int, block_def):
        roll = rnd.random()
        if roll < (1/4):
            #print("SINGLE INPUT")
            mutate_methods.mutate_single_input(indiv, block_index, block_def)
        elif roll < (2/4):
            #print("SINGLE ARG VALUE")
            mutate_methods.mutate_single_argvalue(indiv, block_index, block_def)
        elif roll < (3/4):
            #print("SINGLE ARG INDEX")
            mutate_methods.mutate_single_argindex(indiv, block_index, block_def)
        else:
            #print("SINGLE FTN")
            mutate_methods.mutate_single_ftn(indiv, block_index, block_def)
