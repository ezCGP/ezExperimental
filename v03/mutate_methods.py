'''
required inputs:
 * a single IndividualMaterial (already deepcopied if applicable)
 * a single int for which block index we should mutate

no outputs; the mutate method will alter the given individual. it is expected
that the original individual has already been deepcopied.
'''

# packages
import sys
import os
from numpy import random as rnd
import numpy as np
from copy import deepcopy

# scripts


def mutate_1(indiv: IndividualMaterial, block_index: int):
	pass

def mutate_2(indiv: IndividualMaterial, block_index: int):
	pass


'''
# tried to tuck these all under a class so I can leverage the all() method
but if we don't make an instance of the class... ie mutmeths = MutateMethods()
then we can't use the 'self' value. and it seems slopier to try and make an
instance of this class so I gave up on it...instead just list the methods

class MutateMethods():

	def all(self):
		'''
		#gather all callable methods (excluding 'all'), and return
		'''
		methods = []
		for name, val in type(self).__dict__.items():
			if (callable(val)) and (not 'all'):
				methods.append(name)
		return methods

	def mutate_1(self, indiv: IndividualMaterial, block_index: int):
		pass

	def mutate_2(self, indiv: IndividualMaterial, block_index: int):
		pass
'''