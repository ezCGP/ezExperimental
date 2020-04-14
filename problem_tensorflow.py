# packages
import numpy as np

# scripts
from problem_interface import ProblemDefinition
from factory import TensorFactory
from operators import TFOps
from arguments import NoArgs
from evaluate import IndividualStandardEvaluate, BlockTensorFlowEvaluate
from mutate import InidividualMutateA, BlockMutateA
from mate import IndividualMateA, BlockNoMate

from database.ezDataLoader import load_CIFAR10

# Fitness imports
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score as accuracy

# This is a temporary import. We are forcing normalization since we only have one training block
from pipeline_operators import Normalize

class Problem(ProblemDefinition):
    def __init__(self):
        population_size = 8
        number_universe = 1
        factory = TensorFactory
        factory_instance = factory()
        mpi = True
        super().__init__(population_size, number_universe, factory, mpi)

        block_def = self.construct_block_def(nickname = "main_block",
                                             shape_def = factory_instance.build_shape(),
                                             operator_def =  TFOps,
                                             argument_def = NoArgs,
                                             evaluate_def = BlockTensorFlowEvaluate,
                                             mutate_def = BlockMutateA,
                                             mate_def = BlockNoMate)

        self.construct_individual_def(block_defs = [block_def],
                                    mutate_def = InidividualMutateA,
                                    mate_def = IndividualMateA,
                                    evaluate_def = IndividualStandardEvaluate)

        # where to put this?
        self.construct_dataset()


    def goal_function(self, data):
        # TODO what is this
        return 1/data

    def construct_dataset(self):
        """
        Loads cifar 10
        :return: None
        """
        dataset = load_CIFAR10(.8, .2)

        # force normalization  # will now apply to both pipelines
        dataset.preprocess_pipeline.add_operation(Normalize())

        self.data = dataset

    def objective_functions(self, indiv):
        """

        :param indiv: individual which contains references to output of training
        :return: None
        """
        dataset = self.data
        _, actual = dataset.preprocess_test_data()
        actual = np.argmax(actual, axis = 1)
        predict = indiv.output
        predict = np.argmax(predict, axis = 1)
        acc_score = accuracy(actual, predict)
        f1 = f1_score(actual, predict, average = "macro")
        indiv.fitness.values = (-acc_score, -f1)  # want to minimize this

    def check_convergence(self, universe):
        """

        :param universe:
        :return:
        """
        GENERATION_LIMIT = 1
        SCORE_MIN = 1e-1

        print("\n\n\n\n\n", universe.generation, np.min(np.array(universe.fitness_scores)))

        if universe.generation >= GENERATION_LIMIT:
            print("TERMINATING...reached generation limit")
            universe.converged = True
        if np.min(np.array(universe.fitness_scores)[0]) < SCORE_MIN:
            print("TERMINATING...reached minimum score")
            universe.converged = True