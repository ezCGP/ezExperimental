# packages
from typing import List
from abc import ABC, abstractmethod
from numpy import random as rnd

# scripts



class Mutate_Methods():

	def all(self):
		methods = []
		for name, val in type(self).__dict__.items():
			if (callable(val)) and (not 'all'):
				methods.append(name)
		return methods

	def mutate_1(indiv: Genome_Material, block_i: int):
		# do something to indiv
		# don't need to return indiv...will mutate in place
		pass

	def mutate_2(indiv: Genome_Material, block_i: int):
		# do something to indiv
		pass


class Mutatable(ABC):
	@abstractmethod
	def mutate():
		pass


class Mutate_A(Mutatable):
	def mutate(indiv: Genome_Material, block_i: int):
		# equally assign equal weights to all methods
		all_methods = Mutate_Methods.all(Mutate_Methods)
		roll = rnd.randint(len(all_methods))
		mut_method = all_methods[roll]
		mut_method(indiv, block_i)


class Mutate_B(Mutatable):
	def mutate(indiv: Genome_Material, block_i: int):
		Mutate_Methods.mutate_1(indiv, block_i)


class Mutate_C(Mutatable):
	def mutate(indiv: Genome_Material, block_i: int):
		Mutate_Methods.mutate_2(indiv, block_i)










class Evaluator(ABC):
	@abstractmethod
	def evaluate():
		pass

	def score_fitness():
		print("if you get this then last block-evaluator doesn't have this filled in")


class SymbolicRegressionEval(Evaluator):
	def evaluate(training_datapair: tuple):
		data, _ = training_datapair
		#...evaluate...
		return output

	def score_fitness(training_datapair: tuple, output_data):
		_, labels = training_datapair
		error = output_data - labels
	    rms_error = np.sqrt(np.mean(np.square(error)))
	    max_error = np.max(np.abs(error))
	    return rms_error, max_error


class TensorFlowEval(Evaluator):
	def evaluate(training_datapair: tuple, validation_datapair: tuple):
		train_data, train_labels = training_datapair
		valid_data, _ = validation_datapair
		# ...training...
		# ...run valid_data...
		return validation_output

	def score_fitness(validation_datapair: tuple, output_data):
		_, valid_labels = validation_datapair
		# ...compare with output_data
		return score









class BlockType():
	def __init__(self,
				nickname: str="default",
				input_dtypes: list=[],
				output_dtypes: list=[],
				main_count: int=20,
				arg_count: int=50,
				mutable: Mutable,
				matable: Matable,
				evaluator: Evaluator,
				operators: Operators,
				arguments: Arguments):

		self.nickname = nickname
		self.input_dtypes = input_dtypes
		self.input_count = len(input_dtypes)
		self.output_dtypes = output_dtypes
		self.output_count = len(output_dtypes)
		self.main_count = main_count
		self.genome_count = self.input_count+self.output_count+self.main_count
		self.arg_count = arg_count

		# add interface objects
		self.mutable = mutable
		self.matable = matable
		self.evaluator = evaluator
		self.operators = operators
		self.arguments = arguments

		# fill weights
		#...


	def init_block(self, block: Block_Material):
		block.args = [None]*self.arg_count
		self.fill_args(block)

		block.genome = [None]*self.genome_count
		block.genome[(-1*self.input_count):] = ["InputPlaceholder"]*self.input_count
		self.fill_genome(block)


	def mutate(self, block: Block_Material):
		self.mutable.mutate(block)


	def mate(self, block: Block_Material):
		self.matable.mate(block)


	def evaluate(self, block: Block_Material, training_datapair=None, validation_datapair=None):
		self.evaluator.evaluate(block)


class IndividualType():
	def __init__(self, block_defs: List(BlockType)):
		self.block_defs = block_defs
		self.block_count = len(block_defs)

	def __getitem__(self, index: int):
		return self.block_defs[i]

	def mutate(self, indiv: Individual_Material):
		for i in range(self.block_count):
			self[i].mutate(indiv[i])

	def evaluate(self, indiv: Individual_Material, training_datapair=None, validation_datapair=None):
		for i in range(self.block_count):
			input_data = self[i].evaluate(indiv[i], inputd_data)



class Block_Material():
	def __init__(self, block_def: BlockType):
		block_def.init_block(self) #fills .genome and .args

	def __setitem__(self):
		pass

	def __getitem__(self):
		pass


class Individual_Material():
	def __init__(self, indiv_def: IndividualType):
		self.blocks = []
		for block_def in indiv_def.blocks:
			self.block.append(Block_Material(block_def))

	def __getitem__(self):
		pass



preprocessing_block = BlockType(Mutable_A,...)
tensorflow_block = BlockType(Mutable_B,...)
individual_skeleton = IndividualType([preprocessing_block, tensorflow_block])
population = []
for _ in range(pop_size):
	individual = Individual_Material(IndividualType)
