# packages
import os
import sys
from copy import deepcopy
from abc import ABC, abstractmethod
from numpy import random as rnd


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