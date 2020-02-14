'''
Script to define the unique representation of each individual

And individual is comprised of 'n' number of blocks. So an individual
will be an instance of IndividualMaterial() which itself is primarily
a list of BlockMaterial() instances.
'''

# packages
import sys
import os
import numpy as np

# scripts
from genetic_definition import IndividualDefinition, BlockDefinition


class BlockMaterial():
    '''
    attributes:
     * genome: list of mostly dictionaries
     * args: list of args
     * need_evaluate: boolean flag
     * output: TODO maybe have a place to add the output after it has been evaluated
    '''
    def __init__(self, block_def: BlockDefinition):
        '''
        sets these attributes:
         * need_evaluate = False
         * genome
         * args
         * active_nodes
         * active_args
        '''
        block_def.init_block(self)
        block_def.evaluate_def.reset_evaluation(self) # or move...ya prob move to evaluate

    def __setitem__(self, node_index, value):
        self.genome[node_index] = value

    def __getitem__(self, node_index):
        return self.genome[node_index]


class IndividualMaterial():
    '''
    attributes:
     * blocks: list of BlockMaterial instances
     * fitness: instance of class Fitness which is required for MultiObjective Optimization

    methods:
     * need_evalute: checks the respective boolean flag in all blocks
     and returns True if at least any single block is True
    '''
    def __init__(self, indiv_def: IndividualDefinition):
        self.fitness = self.Fitness()
        self.blocks = []
        for block_def in indiv_def.block_defs:
            self.blocks.append(BlockMaterial(block_def))

    def __setitem__(self, block_index, block: BlockMaterial):
        self.blocks[block_index] = block

    def __getitem__(self, block_index):
        return self.blocks[block_index]

    def need_evaluate(self):
        for block in self.blocks:
            if block.need_evaluate:
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


if __name__ == "__main__":
    '''
    quickly build an individual to check it's footprint
    '''
    block_def = BlockDefinition(...)
    individual_def = IndividualDefinition([block_def])
    individual = IndividualMaterial(individual_def)

    print("with sys...\n", sys.getsizeof(individual))
    try:
        from pympler import asizeof
        print("with pympler...\n", asizeof.asizeof(individual))
        print(asizeof.asized(ting, detail=1).format())
    except ModuleNotFoundError:
        print("module pympler not installed...skipping")