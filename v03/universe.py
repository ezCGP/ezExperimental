'''
words
'''

# packages
import os
import sys
from abc import ABC, abstractmethod


# scripts
from genetic_material import IndividualMaterial
import problem


class Universe(ABC):
    def __init__(self,
                individual_def: IndividualDefinition,
                population_size: int,
                generation_cap: int,
                fitness_cap,
                seed: int = 9):
        self.indiv_def = individual_def
        self.pop_size = population_size
        self.generation_cap = generation_cap
        np.random.seed(seed)


    @abstractmethod
    def evaluate(self, population):
        '''
        describe how to evaluate the population as a whole.
        this is where we would implement multi processing
        '''
        pass
    

    def init_population(self):
        self.generation = 0
        self.converged = False
        population = []
        for i in range(self.pop_size):
            individual = IndividualMaterial(self.indiv_def)
            population.append(individual)
        return population


    def parent_selection(self, population):
        return = tournamentSelection(population, k=len(population))


    def population_selection(self, population, desired_size):
        population, _ selections.selNSGA2(population, k=desired_size, nd='standard')
        return population


    def evolve_population(self, population):
        # MATE
        children = []
        mating_list = self.parent_selection(population)
        for ith_indiv in range(0, len(mating_list), 2):
            parent1 = population[ith_indiv]
            parent2 = population[ith_indiv+1]
            children += self.indiv_def(parent1, parent2)
        population += children

        # MUTATE
        mutants = []
        for individual in population:
            mutants += self.indiv_def.mutate(individual)
        population += mutants

        return population

    def check_convergence(self):
        self.converged = self.generation < self.generation_cap


    def run_universe(self):
        population = self.init_population()
        self.evaluate(population)

        while not self.converged:
            self.generation += 1
            self.evolve_population(population)
            self.evaluate(population)
            self.population_selection(population, self.pop_size)