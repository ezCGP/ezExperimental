# packages
import numpy as np

# scripts
from problem_interface import ProblemDefinition
from factory import Factory
from operators import SymbRegressionNoArgs
from arguments import NoArgs
from evaluate import IndividualStandardEvaluate, BlockStandardEvaluate
from mutate import InidividualMutateA, BlockMutateA
from mate import IndividualMateA, BlockNoMate
from database.data_pair import DataPair


class Problem(ProblemDefinition):
    def __init__(self):
        population_size = 100
        number_universe = 1
        factory = Factory
        factory_instance = factory()
        self.mpi = True
        super().__init__(population_size, number_universe, factory, self.mpi)

        block_def = self.construct_block_def(nickname = "main_block",
                                            shape_def = factory_instance.build_shape(),
                                            operator_def = SymbRegressionNoArgs,
                                            argument_def = NoArgs,
                                            evaluate_def = BlockStandardEvaluate,
                                            mutate_def = BlockMutateA,
                                            mate_def = BlockNoMate)

        self.construct_individual_def(block_defs = [block_def],
                                    mutate_def = InidividualMutateA,
                                    mate_def = IndividualMateA,
                                    evaluate_def = IndividualStandardEvaluate)

        # where to put this?
        self.construct_dataset()


    def goal_function(self, data):
        return 1/data

    def construct_dataset(self):
        x_train = [np.float64(1), np.random.uniform(low=0.25, high=2, size=200)]
        y_train = self.goal_function(x_train[1])
        self.data = DataPair(x_train, y_train)

    def objective_functions(self, indiv):
        x_train, y_train = self.data.get_data()
        actual = y_train

        predit = indiv.output
        error = actual-predit
        rms_error = np.sqrt(np.mean(np.square(error)))
        max_error = np.max(np.abs(error))
        indiv.fitness.values = (rms_error, max_error)

    def check_convergence(self, universe):
        GENERATION_LIMIT = 199
        SCORE_MIN = 1e-1

        print("\n\n\n\n\n", universe.generation, np.min(np.array(universe.fitness_scores)))

        if universe.generation >= GENERATION_LIMIT:
            print("TERMINATING...reached generation limit")
            universe.converged = True
        if np.min(np.array(universe.fitness_scores)[0]) < SCORE_MIN:
            print("TERMINATING...reached minimum score")
            universe.converged = True