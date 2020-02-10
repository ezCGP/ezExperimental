# packages
from numpy import random as rnd

# scripts
from mutate_methods import Mutate_Methods
from interface_mutable import Mutatable

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