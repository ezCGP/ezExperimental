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
	def evaluate(self, block: BlockMaterial, training_datapair, validation_datapair=None):
		pass

	def import_list(self):
		'''
		in theory import packages only if we use the respective EvaluateDefinition

		likely will abandon this
		'''
		return []


class StandardEvaluate(EvaluateDefinition):

	def evaluate(self, block: BlockMaterial, training_datapair, validation_datapair=None):
		pass


class MPIStandardEvaluate(EvaluateDefinition):

	def evaluate(self, block: BlockMaterial, training_datapair, validation_datapair=None):
		pass


class PreprocessEvaluate(EvaluateDefinition):

	def evaluate(self, block: BlockMaterial, training_datapair, validation_datapair=None):
		pass


class TensorFlowEvaluate(EvaluateDefinition):

	def evaluate(self, block: BlockMaterial, training_datapair, validation_datapair):
		pass
