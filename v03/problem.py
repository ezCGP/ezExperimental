'''
words
'''

# packages



# scripts
from genetic_definition import BlockDefinition, IndividualDefinition
from mutate import MutateA
from mate import WholeMateOnly
from evaluate import PreprocessEvaluate, TensorFlowEvaluate
from operator import OperatorA
from argument import ArgumentA
from shape import ShapeA

block_def1 = BlockDefinition("preprocessor",
							ShapeA(),
							MutateA(),
							WholeMateOnly(),
							PreprocessEvaluate(),
							OperatorA(),
							ArgumentA())
block_def2 = BlockDefinition("tf",
							ShapeA(),
							MutateA(),
							WholeMateOnly(),
							TensorFlowEvaluate(),
							OperatorA(),
							ArgumentA())

individual_def = IndividualDefinition([block_def1, block_def2])

POP_SIZE = 100
training_datapair = None
validation_datapair = None