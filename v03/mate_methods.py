'''
attach all mate methods to the MateMethods class

required inputs:
 * 2 parent IndividualMaterial
 * 1 int for the block index we want to mate

required output:
 * it is expected that the parents do not get altered so deepcopies are required
 * list of offspring IndividualMaterial to be added to the population
'''

# packages
import sys
import os
from numpy import random as rnd
import numpy as np
from copy import deepcopy

# scripts

def whole_block(parent1: IndividualMaterial, parent2: IndividualMaterial, block_index: int):
	child1 = deepcopy(parent1)
	child1[block_index] = deepcopy(parent2[block_index])

	child2 = deepcopy(parent2)
	child2[block_index] = deepcopy(parent1[block_index])

	return [child1, child2]

def partial_block(parent1: IndividualMaterial, parent2: IndividualMaterial, block_index: int):
	child1 = deepcopy(parent1)
	child2 = deepcopy(parent2)
	pass

	return [child1, child2]


'''
See MutateMethods() for my note as to why I removed this
class MateMethods():

	def __all__(self):
		'''
		#gather all callable methods (excluding 'all'), and return
		'''
		methods = []
		for name, val in type(self).__dict__.items():
			if (callable(val)) and (not 'all'):
				methods.append(name)
		return methods


	def whole_block(self, parent1: IndividualMaterial, parent2: IndividualMaterial, block_index: int):
		child1 = deepcopy(parent1)
		child1[block_index] = deepcopy(parent2[block_index])

		child2 = deepcopy(parent2)
		child2[block_index] = deepcopy(parent1[block_index])

		return [child1, child2]

	def partial_block(self, parent1: IndividualMaterial, parent2: IndividualMaterial, block_index: int):
		child1 = deepcopy(parent1)
		child2 = deepcopy(parent2)
		pass

		return [child1, child2]
'''