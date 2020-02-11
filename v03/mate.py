'''
don't include self in any of these classes. we don't plan on making instances of the class.
'''

# packages
import sys
import os
from abc import ABC, abstractmethod
from numpy import random as rnd
import numpy as np

# scripts
import mate_methods


class MateDefinition(ABC):
	'''
	words
	'''
	prob_mate = 0.10 # default value

	@abstractmethod
	def mate(parent1: IndividualMaterial, parent2: IndividualMaterial, block_index: int):
		pass



class WholeMateOnly(MateDefinition):
	'''
	each pair of block/parents will mate w/prob 25%

	if they mate, they will only mate with whole_block()
	'''
	prob_mate = 0.25
	def mate(parent1: IndividualMaterial, parent2: IndividualMaterial, block_index: int):
		mate_methods.whole_block(parent1, parent2, block_index)