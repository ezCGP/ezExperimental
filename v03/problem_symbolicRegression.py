# packages
import numpy as np

# scripts
from problem_interface import ProblemDefinition
from factory import Factory
from shape import ShapeA
from operators import SymbRegressionNoArgs
from arguments import NoArgs
from evaluate import IndividualStandardEvaluate
from mutate import BlockMutateA
from mate import BlockNoMate


class SymbolicRegression1(ProblemDefinition):
    def __init__(self):
        population_size = 100
        number_universe = 1
        factory = Factory
        super().__init__(population_size, number_universe, factory)

        block_def = self.construct_block(nickname = "main_block",
                                        shape_def = ShapeA,
                                        operator_def = SymbRegressionNoArgs,
                                        argument_def = NoArgs,
                                        evaluate_def = IndividualStandardEvaluate,
                                        mutate_def = BlockMutateA,
                                        mate_def = BlockNoMate)

        self.indiv_def = self.construct_individual(block_defs = [block_def]):


    def goal_function(self, data):
        return 1/x

    def construct_dataset(self):
        self.x_train = [np.float64(1), np.random.uniform(low=0.25, high=2, size=200)]
        self.y_train = self.goalFunction(self.x_train[1])
        #self.x_test = np.random.uniform(low=0.25, high=2, size=20)
        #self.y_test = self.goalFunction(self.x_test)

    def objective_functions(self, indiv):
        actual = self.y_train
        predit = indiv.output
        error = actual-predict
        rms_error = np.sqrt(np.mean(np.square(error)))
        max_error = np.max(np.abs(error))
        indiv.fitness.values = (rms_error, max_error)

    def check_convergence(self, universe):
        GENERATION_LIMIT = 199
        SCORE_MIN = 1e-1
        if universe.generation >= GENERATION_LIMIT:
            print("TERMINATING...reached generation limit")
            universe.converged = True
        if np.min(np.array(universe.fitness_scores)[0]) < SCORE_MIN:
            print("TERMINATING...reached minimum score")
            universe.converged = True