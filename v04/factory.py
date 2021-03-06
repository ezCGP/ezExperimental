'''
factory class will be tasked with building/__init__ all the other classes.
that way we can wrap other debugging + logging items around the init of each class
'''

# packages
import os
import sys
from abc import ABC, abstractmethod


# scripts
from genetic_material import IndividualMaterial, BlockMaterial
from genetic_definition import IndividualDefinition, BlockDefinition
from population import PopulationDefinition, SubPopulationDefinition


class Factory():
    def __init__(self):
        pass

    def build_population(self, indiv_def: IndividualDefinition, population_size):
        my_population = PopulationDefinition(population_size)
        for _ in range(population_size):
            indiv = self.build_individual(indiv_def)
            my_population.population.append(indiv)
        return my_population

    def build_subpopulation(self, indiv_def: IndividualDefinition, population_size, number_subpopulation=None, subpopulation_size=None):
        my_population = SubPopulationDefinition(population_size, number_subpopulation, subpopulation_size)
        for ith_subpop, subpop_size in my_population.subpop_size:
            for _ in range(subpop_size):
                indiv = self.build_individual(indiv_def)
                my_population[ith_subpop].append(indiv)
        return my_population


    def build_individual(self, indiv_def: IndividualDefinition):
        indiv = IndividualMaterial()
        for block_def in indiv_def.block_defs:
            block = self.build_block(block_def)
            indiv.blocks.append(block)
        return indiv


    def build_block(self, block_def: BlockDefinition):
        block = BlockMaterial()
        block_def.init_block(block)
        block_def.evaluate_def.reset_evaluation(block) # or move...ya prob move to evaluate TODO
        return block



