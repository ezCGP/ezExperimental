# packages
import numpy as np

# scripts
from problem_interface import ProblemDefinition
from factory import TensorFactory
from operators import TFOps, PreprocessingOps, AugmentationOps
from arguments import TFArgs, NoArgs, AugmentArgs
from evaluate import IndividualStandardEvaluate, BlockAugmentationEvaluate, BlockPreprocessEvaluate, \
    BlockTensorFlowEvaluate
from mutate import InidividualMutateA, BlockMutateA
from mate import IndividualMateA, BlockNoMate
from shape import ShapeAugmentor, ShapeTensor

from database.ezDataLoader import load_CIFAR10

# Fitness imports
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score as accuracy
from shape import ShapeTensor


class Problem(ProblemDefinition):
    def __init__(self):
        population_size = 8
        number_universe = 1
        factory = TensorFactory
        mpi = True
        super().__init__(population_size, number_universe, factory, mpi)

        augmentation_block_def = self.construct_block_def(nickname="augmentation_block",
                                                          shape_def=ShapeAugmentor,
                                                          operator_def=AugmentationOps,
                                                          argument_def=AugmentArgs,
                                                          evaluate_def=BlockAugmentationEvaluate,
                                                          mutate_def=BlockMutateA,
                                                          mate_def=BlockNoMate)

        preprocessing_block_def = self.construct_block_def(nickname="preprocessing_block",
                                                           shape_def=ShapeAugmentor,
                                                           operator_def=PreprocessingOps,
                                                           argument_def=NoArgs, # no arguments for preprocessing
                                                           evaluate_def=BlockPreprocessEvaluate,
                                                           mutate_def=BlockMutateA,
                                                           mate_def=BlockNoMate)

        tensorflow_block_def = self.construct_block_def(nickname="tensorflow_block",
                                                        shape_def=ShapeTensor,
                                                        operator_def=TFOps,
                                                        argument_def=NoArgs,
                                                        evaluate_def=BlockTensorFlowEvaluate,
                                                        mutate_def=BlockMutateA,
                                                        mate_def=BlockNoMate)

        self.construct_individual_def(
            block_defs=[augmentation_block_def, preprocessing_block_def, tensorflow_block_def],
            mutate_def=InidividualMutateA,
            mate_def=IndividualMateA,
            evaluate_def=IndividualStandardEvaluate)

        # where to put this?
        self.construct_dataset()

    def construct_dataset(self):
        """
        Loads cifar 10
        :return: None
        """
        dataset = load_CIFAR10(.8, .2)
        self.data = dataset

    def objective_functions(self, indiv):
        """

        :param indiv: individual which contains references to output of training
        :return: None
        """
        dataset = self.data
        _, actual = dataset.preprocess_test_data()
        actual = np.argmax(actual, axis=1)
        predict = indiv.output
        predict = np.argmax(predict, axis=1)
        acc_score = accuracy(actual, predict)
        f1 = f1_score(actual, predict, average="macro")
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
