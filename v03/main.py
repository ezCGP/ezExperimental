'''

'''

# packages
import os
import sys
import numpy as np

# scripts
from universe import MPIUniverse, Universe


if __name__ == "__main__":
    import argparse
    parser.argparse.ArgumentParser()
    parser.add_argument("-p", "--problem",
                        type = str,
                        required = True,
                        help = "pick which problem class to import")
    parser.add_argument("-s", "--seed",
                        type = int,
                        default = 0,
                        help = "pick which seed to use for numpy")
    args = parser.parse_args()



    # figure out which problem py file to import
    if args.problem.endswith('.py'):
        args.problem = args.problem[:-3]
    problem_module = __import__(args.problem)

    problem = problem_module.Problem()
    for ith_universe in problem.number_universe:
        # set the seed
        np.random.seed(args.seed + ith_universe)

        # init corresponding universe
        if problem.mpi:
            universe = MPIUniverse(problem)
        else:
            universe = Universe(problem)

        # run
        universe.run(population, problem)