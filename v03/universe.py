'''
words
'''

# packages
import os
import sys
from abc import ABC, abstractmethod
import time
import numpy as np

# scripts
from genetic_material import IndividualMaterial
from problem_interface import ProblemDefinition
from population import PopulationDefinition
from typing import List

from mpi4py import MPI




class Universe():
    def __init__(self, problem: ProblemDefinition):
        '''
        TODO:
        should we try and pass off problem-class attributes to universe???
        or just call from problem as we need... ex: problem.indiv_def
        '''
        self.factory = problem.Factory()
        self.population = self.factory.build_population(problem.indiv_def, problem.pop_size)
        self.problem = problem
        self.converged = False


    """
    1. Where tournament Selection from?
    """
    def parent_selection(self):
        return tournamentSelection(self.population.population, k=len(self.population.population))


    def evolve_population(self, problem: ProblemDefinition):
        # MATE
        children = []
        mating_list = self.parent_selection()
        for ith_indiv in range(0, len(mating_list), 2):
            parent1 = mating_list[ith_indiv]
            parent2 = mating_list[ith_indiv+1]
            children += problem.indiv_def.mate(parent1, parent2)
        self.population.add_next_generation(children)

        # MUTATE
        mutants = []
        for individual in self.population:
            mutants += problem.indiv_def.mutate(individual)
        self.population.add_next_generation(mutants)


    def evaluate_score_population(self, problem: ProblemDefinition):
        self.fitness_scores = []
        for indiv in self.population.population:
            # EVALUATE
            problem.indiv_def.evaluate(indiv, problem.dataset)
            # SCORE
            problem.objective_functions(indiv)

    """
    2. Under poulation_selection, where selections from?
    """
    def population_selection(self):
        self.population.population, _ = selections.selNSGA2(self.population.population, self.population.pop_size, nd='standard')
        return 0 # return array of sub-pops


    def check_convergence(self, problem_def: ProblemDefinition):
        '''
        Should update self.converged
        '''
        problem_def.check_convergence(self)



    def run(self, problem: ProblemDefinition):
        '''
        assumes a population has only been created and not evaluatedscored
        '''

        self.evaluate_score_population(problem)
        while not self.converged:
            self.evolve_population(self.population)
            self.evaluate_score_population(problem)
            self.population_selection(self.population)
            self.converged = self.check_convergence(problem)

class MPIUniverse(Universe):
    def __init__(self, problem: ProblemDefinition):
        '''
        TODO:
        should we try and pass off problem-class attributes to universe???
        or just call from problem as we need... ex: problem.indiv_def
        '''
        super().__init__(problem)

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
        size = comm.Get_size() # number of CPUs
        rank = comm.Get_rank() # this CPU's rank
        print('Start MPI Universe Run')

        seed = problem.SEED
        np.random.seed()
        
        # DatasetObject pass it into evaluation
        
        if rank == 0:
            split_population = self.population.split(size)
        else:
            split_population = []
        converged = False
        while not converged:
            
            self.population = comm.scatter(split_population, root=0)
            # evolve and evaluate
            self.evolve_population(problem)
            self.evaluate_score_population(problem)
            
            comm.Barrier()
            split_population = comm.gather(self.population, root=0)
            if rank == 0:
                # population_selection should merge a lsit of pop objects, perform selection, split it into an array of pop objects and return that array
                split_population = self.population_selection()
                self.check_convergence(problem)
            
            self.converged = comm.bcast(self.converged, root=0)
        # check convergence takes care of gen limit too
        # merge self.population array of individuals
        self.population = util.merge_population(split_population)
        self.save_results()
    
    def save_results(self):
        pass

    def population_selection_mpi(self, sub_pops: List[PopulationDefinition]):
        """
        1. Merge sub-pops
        2. Run pop_selection
        3. Split into sub-pops
        """
        pass














