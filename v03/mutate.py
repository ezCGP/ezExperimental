'''
words
'''

# packages
import os
import sys
from abc import ABC, abstractmethod
from numpy import random as rnd

# scripts
import mutate_methods


class MutateDefinition(ABC):
	'''
	made into an abstract class

	anything with @abstractmethod will have to be filled in by
	whatever class that inherits from MutateDefinition
	'''
	prob_mutate = 0.30 # default

	@abstractmethod
	def mutate(indiv: IndividualMaterial, block_index: int):
		pass


class MutateA(MutateDefinition):
	'''
	each block has a prob of mutate at 10%

	then if we choose to mutate that block,
	it will use muatate_1 25% of the time and
	mutate_2 75% of the time
	'''
	prob_mutate = 0.10
	def mutate(indiv: IndividualMaterial, block_index: int):
		roll = rnd.random()
		if roll < 0.25:
			mutate_methods.mutate_1(indiv, block_index)
		else:
			mutate_methods.mutate_2(indiv, block_index)