'''
words
'''

# packages
from copy import deepcopy


# scripts
from mutate import MutateB
from mate import WholeMateOnly
from evaluate import StandardEvaluate
from operators import SymbRegressionNoArgs
from arguments import NoArgs
from shape import ShapeA
from genetic_definition import BlockDefinition, IndividualDefinition
from genetic_material import IndividualMaterial

block_def1 = BlockDefinition("preprocessor",
							ShapeA(),
							MutateB(),
							WholeMateOnly(),
							StandardEvaluate(),
							SymbRegressionNoArgs(),
							NoArgs())


POP_SIZE = 1
individual_def = IndividualDefinition([block_def1])
ind = IndividualMaterial(individual_def)
#mut = deepcopy(ind)
#individual_def.mutate(mut)
muts = individual_def.mutate(ind)

pop = [ind]+muts
for j, person in enumerate(muts):
	print("MUTANT %i" % j)
	for i, (onode, mnode) in enumerate(zip(ind[0].genome,person[0].genome)):
		if onode != mnode:
			if i in ind[0].active_nodes:
				print(i, "active node")
			else:
				print(i, "node")
			print(onode)
			print(mnode)

	for i, (onode, mnode) in enumerate(zip(ind[0].args,person[0].args)):
		if onode != mnode:
			if i in ind[0].active_args:
				print(i, "active arg")
			else:
				print(i, "arg")
			print(onode)
			print(mnode)

	print("")

print("try mating")
offspring = individual_def.mate(muts[0],muts[1])

offspring[1][0].genome = muts[0][0].genome 