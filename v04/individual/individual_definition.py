# packages
import sys
import os
from typing import List
from numpy import random as rnd
import numpy as np
from copy import deepcopy

#import individual_mutate #won't work

#import individual.individual_mutate #works
#ting = individual.individual_mutate.MutateDefinition()

#from . import individual_mutate
#ting = individual_mutate.InidividualMutateA()

import v04.individual.individual_mutate
ting = v04.individual.individual_mutate.InidividualMutateA()

class IndividualDefinition():
    def __init__(self,
                block_defs,
                mutate_def,
                mate_def,
                evaluate_def):
        self.block_defs = block_defs
        self.block_count = len(block_defs)
        self.mutate_def = mutate_def()
        self.mate_def = mate_def()
        self.evaluate_def = evaluate_def()

    def __getitem__(self, block_index: int):
        return self.block_defs[block_index]


    def get_actives(self, indiv):
        for block_index, block in enumerate(indiv.blocks):
            self[block_index].get_actives(indiv[block_index])

    def mutate(self, indiv):
        mutants = self.mutate_def.mutate(self, indiv)
        return mutants

    def mate(self, parent1, parent2):
        children = self.mate_def.mate(self, parent1, parent2)
        return children

    def evaluate(self, indiv, training_datapair, validation_datapair=None):
        self.evaluate_def.evaluate(self, indiv, training_datapair, validation_datapair)