

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

	def mutate_2(indiv: Genome_Material, block_i: int):
		# do something to indiv