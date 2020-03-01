'''

'''

# packages
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List


# scripts
from shape import ShapeMetaDefinition
from operators import OperatorDefinition
from arguments import ArgumentDefinition
from evaluate import EvaluateDefinition
from mutate import MutateDefinition
from mate import MateDefinition

from genetic_definition import BlockDefinition, IndividualDefinition
from genetic_material import IndividualMaterial

from factory import Factory
#from universe import Universe

class ProblemDefinition(ABC):
    '''
     * data: training + validation
     * objective ftn(s)
     * define convergence
     * individual definition
     * which universe
    '''
    # Note to user: all abstract methods are not defined 
    # by default, please implement according to preferences
    def __init__(self,
                population_size,
                number_universe,
                factory_def: Factory,
                mpi = False):
        '''
        self.construct_dataset()

        the build out each block and build individual_def
        block_def = self.construct_block()
        self.construct_individual([block_def])
        '''
        self.pop_size = population_size
        self.number_universe = number_universe
        self.Factory = factory_def
        self.mpi = mpi


    @abstractmethod
    def construct_dataset(self):
        '''
        training data + labels
        validating data + labels
        '''
        pass


    @abstractmethod
    def objective_functions(self, indiv: IndividualMaterial):
        '''
        save fitness for each individual to IndividualMaterial.fitness.values as tuple
        
        try:
            acc_score = accuracy_score(actual, predict)
            avg_f1_score = f1_score(actual, predict, average='macro')
            return 1 - acc_score, 1 - avg_f1_score
        except ValueError:
            print('Malformed predictions passed in. Setting worst fitness')
            return 1, 1  # 0 acc_score and avg f1_score b/c we want this indiv ignored
        '''
        pass


    @abstractmethod
    def check_convergence(self, universe):
        '''
        whether some threshold for fitness or some max generation count

        set universe.converged to boolean T/F ...True will end the universe run
        '''
        pass


    # Note to user: these last two methods are already defined
    def construct_block_def(self,
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


    def construct_individual_def(self,
                            block_defs: List, # List(BlockDefinition)
                            mutate_def: MutateDefinition,
                            mate_def: MateDefinition,
                            evaluate_def: EvaluateDefinition):
        self.indiv_def = IndividualDefinition(block_defs,
                                            mutate_def,
                                            mate_def,
                                            evaluate_def)


