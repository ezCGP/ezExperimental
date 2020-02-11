'''
words
'''

# packages
import sys
import os
from abc import ABC, abstractmethod


# scripts



class EvaluateDefinition(ABC):
	'''
	words
	'''

	@abstractmethod
	def evaluate(block: BlockMaterial, training_datapair, validation_datapair=None):
		pass

	def import_list():
		return []


class StandardEvaluate(EvaluateDefinition):

	def evaluate(block: BlockMaterial, training_datapair, validation_datapair=None):
		pass


class MPIStandardEvaluate(EvaluateDefinition):

	def evaluate(block: BlockMaterial, training_datapair, validation_datapair=None):
		pass

	def import_list():
		return ['mpi'] # TODO, how to add this to global?


class PreprocessEvaluate(EvaluateDefinition):

	def evaluate(block: BlockMaterial, training_datapair, validation_datapair=None):
		pass


class TensorFlowEvaluate(EvaluateDefinition):

	def evaluate(block: BlockMaterial, training_datapair, validation_datapair):
		pass
