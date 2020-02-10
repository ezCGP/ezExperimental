class Block_Declaration():

	def __init__(self, skeleton: Block_Definition):
		
		# Inputs
		self.input_dtypes = skeleton['input_dtypes']
		self.input_count = len(self.input_dtypes)

		# Outputs
		self.output_dtypes = skeleton['output_dtypes']
		self.output_count = skeleton[self.output_dtypes]

		# Main
		self.arg_count = skeleton['arg_count']
		self.main_count = skeleton['main_count']
		self.genome_count = self.main_count + self.input_count + self.output_dtypes

		self.buildWeights('arg_methods', skeleton)
		self.buildWeights('ftn_methods', skeleton)
		self.buildWeights('mut_methods', skeleton)
		self.buildWeights('mate_methods', skeleton)




	def init_block(self, indiv: Block_Material, skeleton: Block_Definition):

		# args
		indiv.args = [None]*self.arg_count
		self.fillArgs(indiv)

		# genome
		indiv.genome = [None]*self.genome_count
		indiv.genome[(-1*self.input_count):] = ["InputPlaceholder"]*self.input_count


	def fillArgs(self, indiv: Block_Material):
		pass


	def fillGenome(self, indiv: Block_Material):
		pass


	def buildWeights(self, skeleton: Block_Definition):
		pass