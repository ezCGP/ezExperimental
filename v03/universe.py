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




class Universe():
    def __init__(self,
                problem: ProblemDefinition):
        '''
        TODO:
        should we try and pass off problem-class attributes to universe???
        or just call from problem as we need... ex: problem.indiv_def
        '''
        self.factory = problem.Factory()


    def parent_selection(self):
        return = tournamentSelection(self.population.population, k=len(self.population.population))


    def evolve_population(self, problem):
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
        for individual in population:
            mutants += prblem.indiv_def.mutate(individual)
        self.population.add_next_generation(mutants)


    def evaluate_score_population(self, problem: ProblemDefinition):
        self.fitness_scores = []
        for indiv in self.population.population:
            # EVALUATE
            problem.indiv_def.evaluate(indiv, problem.x_train)
            # SCORE
            problem.objective_functions(indiv)
            self.fitness_scores.append(indiv.fintess.values)


    def population_selection(self):
        self.population.population, _ = selections.selNSGA2(self.population.population, self..population.pop_size, nd='standard')


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
        self.population = self.factory.build_population(problem.indiv_def, problem.pop_size)
        self.generation = 0
        self.evaluate_score_population(problem)
        while not self.converged:
            self.generation += 1
            self.evolve_population(population)
            self.evaluate_score_population(problem)
            self.population_selection(population)
            self.check_convergence(problem)















