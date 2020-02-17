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
    # Note to user: all abstract methods are not defined 
    # by default, please implement according to preferences
    def __init__(self,
                GEN_LIMIT=100,
                POP_SIZE = 20
                N_EPOCHS = 10
                SEED = 17
                N_UNIVERSE = 1
                N_MUTANTS = 2
                N_OFFSPRING = 2 # THIS COMES IN PAIRS (e.g. N_OFFPSRING = 2 is 4/gen)
                MIN_SCORE = 0.00  # terminate immediately when 100% accuracy is achieved

                # Logistics Parameters
                SEED_ROOT_DIR = 'sam_ezCGP_runs/run_20'
                DATASET_NAME = 'cifar10'
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
        print('Train data shape: ' + str(x_train.shape))
        print('Train labels shape: ' + str(y_train.shape))
        print('Validation data shape: ' + str(x_val.shape))
        print('Validation labels shape: ' + str(y_val.shape))
        print('Test data shape: ' + str(x_test.shape))
        print('Test labels shape: ' + str(y_test.shape))
        pass


    @abstractmethod
    def objective_functions(self, population: Population):
        '''
        save fitness for each individual to IndividualMaterial.fitness.values as tuple
        '''
        try:
            acc_score = accuracy_score(actual, predict)
            avg_f1_score = f1_score(actual, predict, average='macro')
            return 1 - acc_score, 1 - avg_f1_score
        except ValueError:
            print('Malformed predictions passed in. Setting worst fitness')
            return 1, 1  # 0 acc_score and avg f1_score b/c we want this indiv ignored



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

    # Note to user: these last two methods are already defined
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


