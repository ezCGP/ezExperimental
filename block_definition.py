

class Block_Definition():
	'''
	still not sure how to handle operators + args
	'''
	def __init__(self,
				nickname: str="default",
				input_dtypes: list=[],
				output_dtypes: list=[],
				main_count: int=20,
				arg_count: int=50,
				mutable: Mutable,
				matable: Matable,
				evaluator: Evaluator):

		self.nickname = nickname
		self.input_dtypes = input_dtypes
		self.input_count = len(input_dtypes)
		self.output_dtypes = output_dtypes
		self.output_count = len(output_dtypes)
		self.main_count = main_count
		self.genome_count = self.input_count+self.output_count+self.main_count
		self.arg_count = arg_count

		self.mutable = mutable
		self.matable = matable
		self.evaluator = evaluator


class Individual_Definition():
	
	def __init__(self,
				#input_dtypes: list=[], # these aren't necc since first and last block will also have this info
				#output_dtypes: list=[],
				block_defs: list=[]):
		# is this necc?
		self.blocks = block_defs
		self.block_count = len(block_defs)