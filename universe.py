'''
words
'''

# packages
import os
import sys
from abc import ABC, abstractmethod
import numpy as np
from tempfile import TemporaryFile

# scripts
from genetic_material import IndividualMaterial
from problem_interface import ProblemDefinition
from utilities_gp.selections import selNSGA2, selTournamentDCD #TODO decide how to do imports later
from genetic_material import IndividualMaterial
from problem_interface import ProblemDefinition
from population import PopulationDefinition
from typing import List


from mpi4py import MPI



class Universe():
    def __init__(self,
                problem: ProblemDefinition,
                output_folder):
        '''
        electing to keep Problem class separate from Universe class...
        if we ever need a Problem attribute, just pass in the instance of Problem as an arguement
        ...as opposed to having a copy of the Problem instance as an attribute of Universe
        '''
        self.factory = problem.Factory()
        self.population = self.factory.build_population(problem.indiv_def, problem.pop_size)
        self.output_folder = output_folder
        self.converged = False


    def parent_selection(self):
        return selTournamentDCD(self.population.population, k=len(self.population.population))


    def evolve_population(self, problem):
        # MATE
        children = []
        mating_list = self.parent_selection() # produce 4
        for ith_indiv in range(0, len(mating_list), 2):
            parent1 = mating_list[ith_indiv]
            parent2 = mating_list[ith_indiv+1]
            children += problem.indiv_def.mate(parent1, parent2)
        self.population.add_next_generation(children)

        # MUTATE
        mutants = []
        for individual in self.population.population:
            mutants += problem.indiv_def.mutate(individual)
        self.population.add_next_generation(mutants)


    def evaluate_score_population(self, problem: ProblemDefinition):
        self.fitness_scores = []
        for indiv in self.population.population:
            # EVALUATE
            problem.indiv_def.evaluate(indiv, problem.data)  # dataset class?
            # SCORE
            problem.objective_functions(indiv)
            self.fitness_scores.append(indiv.fitness.values)


    def population_selection(self):
        self.population.population, _ = selNSGA2(self.population.population, self.population.pop_size, nd='standard')


    def check_convergence(self, problem_def: ProblemDefinition):
        '''
        will assign self.convergence
        '''
        problem_def.check_convergence(self)



    def run(self,
            problem: ProblemDefinition):
        '''
        assumes a population has only been created and not evaluatedscored
        '''
        self.generation = 0
        self.evaluate_score_population(problem)
        self.population_selection()
        while not self.converged:
            self.generation += 1
            self.evolve_population(problem)
            self.evaluate_score_population(problem)
            self.population_selection()
            self.check_convergence(problem)
            self.save_results()

    def save_results(self):

        try:
            file_pop = open('{}/gen{}_pop.npy'.format(self.output_folder, self.generation))
            np.save(file_pop, self.population)

        except IOError: # create a new universe's individuals
            print('Tried to load previous generations, but no files found.')

            newpath = r'{}/'.format(self.output_folder)

            if not os.path.exists(newpath):
                os.makedirs(newpath)

            print(newpath)
            file_pop = '{}/gen{}_pop.npy'.format(newpath, self.generation)
            np.save(file_pop, self.population.population)




# TODO: no input params for run()
class MPIUniverse(Universe):
    def __init__(self, problem: ProblemDefinition, output_folder):
        '''
        TODO:
        should we try and pass off problem-class attributes to universe???
        or just call from problem as we need... ex: problem.indiv_def
        '''
        self.generation = 0
        super().__init__(problem, output_folder)

    def run(self, problem: ProblemDefinition):
        """
        1. Split Population into subpopulation such that number of sup-pops == number of CPUs
        Loop:
        2. Scatter Sub-population from CPU 0 to each of the cpu
        3. Evolve Sub-Population on each CPU
        4. Evaluate Sub-population on each CPU
        5. Gather all the sub-population to CPU 0 (Master CPU)
        6. Perform MPI population selection on CPU 0
            - This produce a new array of sub-population
        7. Check convergence on CPU 0
        8. Broadcast Convergence status to all CPUs
        Repeat Step 2
        """
        comm = MPI.COMM_WORLD
        size = comm.Get_size()  # number of CPUs
        rank = comm.Get_rank()  # this CPU's rank

        # seed = problem.SEED
        # np.random.seed()

        # DatasetObject pass it into evaluation
        print("running")
        self.population = self.factory.build_population(problem.indiv_def, int(problem.pop_size / size))
        self.evaluate_score_population(problem)
        self.population_selection()

        self.population.population = comm.gather(self.population.population, root=0)

        while not self.converged:
            self.population.population = comm.scatter(self.population.population, root=0)
            # evolve and evaluate
            # TODO: if each core mates within own sub-pop, is that ok compared to mating within the entire population?
            # TODO: if the implementation of splitting and merging is too complicated, we can consider evolving one whole pop on CPU0 (shouldn't be too expensive)
            self.evolve_population(problem)
            self.evaluate_score_population(problem)

            comm.Barrier()
            self.population.population = comm.gather(self.population.population, root=0)
            if rank == 0:
                # population_selection should merge a lsit of pop objects, perform selection, split it into an array of pop objects and return that array
                self.population.merge_pop()
                self.population_selection()
                problem.check_convergence(self)
                self.population.split(size)

            self.converged = comm.bcast(self.converged, root=0)

        # check convergence takes care of gen limit too
        # merge self.population array of individuals
        # self.population = util.merge_population(split_population)
        self.save_results()

    def save_results(self):
        file_pop = '{}/gen{}_pop.npy'.format(self.output_folder, self.generation)
        np.save(file_pop, self.population)
        np.save(file_generation, self.generation)
        print("hi")

    def population_selection_mpi(self, sub_pops: List[PopulationDefinition]):
        """
        1. Merge sub-pops
        2. Run pop_selection
        3. Split into sub-pops
        """
        pass