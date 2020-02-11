'''
words
'''

# packages



# scripts
from genetic_material import IndividualMaterial
import problem

population = []
for _ in problem.POP_SIZE:
	individual = IndividualMaterial(problem.individual_def)
	problem.individual_def.evaluate(individual, problem.training_datapair, problem.validation_datapair)
	population.append(individual)