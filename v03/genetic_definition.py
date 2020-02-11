'''
The genetic material of each individual will vary but the structural components will be the same.
This structure is defined in layers:

      STRUCTURE      |||||      DEFINED BY
individual structure |||||  defined by list of blocks
block structure      |||||  defined by shape/meta data, mate methods,
                     |||||    mutate methods, evaluate method, operators
                     |||||    or primitives, argument datatypes
'''

# packages
import sys
import os
from typing import List
from numpy import random as rnd
from copy import deepcopy

# scripts
from mutate import MutateDefinition
from mate import MateDefinition
from evaluate import EvaluateDefinition
from operator import OperatorDefinition
from argument import ArgumentDefinition


class MetaDefinition():
	'''
	a lot of this is just to help fill attributes of a block
	like number of nodes, acceptable input/output datatypes, etc
	'''
	def __init__(self,
				input_dtypes: list=[],
				output_dtypes: list=[],
				main_count: int=20,
				arg_count: int=50):
		self.input_dtypes = input_dtypes
		self.input_count = len(input_dtypes)
		self.output_dtypes = output_dtypes
		self.output_count = len(output_dtypes)
		self.main_count = main_count
		self.genome_count = self.input_count+self.output_count+self.main_count
		self.arg_count = arg_count


class BlockDefinition():
	def __init__(self,
				nickname: str="default",
				meta_def: MetaDefinition,
				mutate_def: MutateDefinition,
				mate_def: MateDefinition,
				evaluate_def: EvaluateDefinition,
				operator_def: OperatorDefinition,
				argument_def: ArgumentDefinition):
		import_list = []
		# Meta:
		self.nickname = nickname
		for name, val in meta_def.__dict__.items():
			# quick way to take all attributes and add to self
			self.__dict__[name] = val
		# Mutate:
		self.mutate_def = mutate_def
		# Mate:
		self.mate_def = mate_def
		# Evaluate:
		self.evaluate_def = evaluate_def
		import_list += evaluate_def.import_list()
		# Operator:
		self.operator_def = operator_def
		# Argument:
		self.argument_def = argument_def


	def init_block(self, block: BlockMaterial):
		'''
		define:
		 * block.genome
		 * block.args
		 * block.need_evaluate
		'''
		block.need_evaluate = True
		block.output = None
		# args:
		block.args = [None]*self.arg_count
		self.fill_args(block)
		# genome:
		block.genome = [None]*self.genome_count
		block.genome[(-1*self.input_count):] = ["InputPlaceholder"]*self.input_count
		self.fill_genome(block)

	def fill_args(self, block: BlockMaterial):
		pass

	def fill_genome(self, block: BlockMaterial):
		pass

	def mutate(self, indiv: IndividualMaterial, block_index: int):
		self.mutate_def.mutate(indiv, block_index)

	def mate(self, parent1: IndividualMaterial, parent2: IndividualMaterial, block_index: int):
		children: List(IndividualMaterial) = self.mate_def.mate(parent1, parent2, block_index)
		return children

	def evaluate(self, block: BlockMaterial, training_datapair, validation_datapair=None):
		output = self.evaluate_def.evaluate(block, training_datapair, validation_datapair)
		# block.output = output # TODO?
		return output



class IndividualDefinition():
	def __init__(self,
				block_defs: List[BlockDefinition]):
		self.block_defs = block_defs
		self.block_count = len(block_defs)

	def __getitem__(self, block_index: int):
		return self.block_defs[i]


	def mutate(self, indiv: IndividualMaterial):
		'''
		right now an individual's block get's mutated with a certain probability

		each block mutatation builds a new mutant respectively...rather than multiple mutations in 1 mutant
		'''
		mutants = []
		for block_index in range(self.block_count):
			roll = rnd.random()
			if roll < self[block_index].mutate_def.prob_mutate:
				mutant = deepcopy(indiv)
				self[block_index].mutate(mutant, block_index)
				mutant[block_index].need_evaluate = True # WARNING: assumption here is that mutate will always change an active node
				mutants.append(mutant)
		return mutants

	def mate(self, parent1: IndividualMaterial, parent2: IndividualMaterial):
		'''
		right now 2 parents have a block mated with a certain probability

		each mated block builds a new child
		'''
		all_children = []
		for block_index in range(self.block_count):
			roll = rnd.random()
			if roll < self[block_index].mate_def.prob_mate:
			children: List(IndividualMaterial) = self.mate_def.mate(parent1, parent2, block_index)
			# for each child, we need to set need_evaluate on all nodes from the mated block and on
			for child in children:
				for block_i in range(block_index, self.block_count):
					child[block_i].need_evaluate = True
			all_children += children #join the lists
		return all_children

	def evaluate(self, indiv: IndividualMaterial, training_datapair, validation_datapair=None):
		for block_index, block in enumerate(indiv.blocks):
			if block.need_evaluate:
				training_datapair = self[block_index].evaluate(block, training_datapair, validation_datapair)