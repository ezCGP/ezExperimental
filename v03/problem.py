'''

'''

# packages
from abc import ABC, abstractmethod
from copy import deepcopy


# scripts
from shape import ShapeMetaDefinition
from operators import OperatorDefinition
from arguments import ArgumentDefinition
from evaluate import EvaluateDefinition
from mutate import MutateDefinition
from mate import MateDefinition

from genetic_definition import BlockDefinition, IndividualDefinition
from genetic_material import IndividualMaterial


POP_SIZE = 1
individual_def = IndividualDefinition([block_def1])


class ProblemDefinition(ABC):
    '''
     * data: training + validation
     * objective ftn(s)
     * define convergence
     * individual definition
     * which universe
    '''

    def __init__(self,
                universe_def,
                population_def):
        '''
        self.construct_dataset()

        the build out each block and build individual_def
        block_def = self.construct_block()
        self.construct_individual([block_def])
        '''
        self.universe_def = universe_def
        self.population_def = population_def


    @abstractmethod
    def construct_dataset(self):
        '''
        training data + labels
        validating data + labels
        '''
        pass


    @abstractmethod
    def objective_functions(self, population: Population):
        '''
        save fitness for each individual to IndividualMaterial.fitness.values as tuple
        '''
        pass


    @abstractmethod
    def check_convergence(self, universe: Universe):
        '''
        whether some threshold for fitness or some max generation count

        set universe.converged to boolean T/F ...True will end the universe run
        '''
        pass


    @abstractmethod
    def build_population(self):
        '''
        to be called in main.py to return an initialized Population() object
        '''
        pass


    def construct_block(self,
                        nickname: str,
                        shape_def: ShapeMetaDefinition,
                        operator_def: OperatorDefinition,
                        argument_def: ArgumentDefinition,
                        evaluate_def: EvaluateDefinition,
                        mutate_def: MutateDefinition,
                        mate_def: MateDefinition):
        return BlockDefinition(nickname,
                            shape_def,
                            operator_def,
                            argument_def,
                            evaluate_def,
                            mutate_def,
                            mate_def)


    def construct_individual(self,
                            block_defs: List(BlockDefinition)):
        self.indiv_def = IndividualDefinition(block_defs)