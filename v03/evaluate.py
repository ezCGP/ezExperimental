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
    REQUIREMENTS/EXPECTATIONS

    Individual Evaluate class:
     * inputs: instance of IndividualDefinition, an instance of IndividualMaterial, and the training+validation data
     * returns: the direct output of the last block

    Block Evaluate class:
     * should start with a reset_evaluation() method
     * inputs: an instance of BlockMaterial, and the training+validation data
     * returns: a list of the output to the next block or as output of the individual
    '''
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, block, training_datapair, validation_datapair=None):
        pass

    @abstractmethod
    def reset_evaluation(self, block):
        pass

    def import_list(self):
        '''
        in theory import packages only if we use the respective EvaluateDefinition

        likely will abandon this
        '''
        return []



class GraphEvaluateDefinition(EvaluateDefinition):
    '''
    attempt at abstracting what an EvaluateDefinition will look like for a 
    computational graph block like tensorflow, pytorch, or keras

    these are just ideas
    '''
    @abstractmethod
    def build_graph(self):
        pass

    @abstractmethod
    def reset_graph(self):
        pass

    @abstractmethod
    def train_graph(self):
        pass

    @abstractmethod
    def run_graph(self):
        pass




class IndividualStandardEvaluate(EvaluateDefinition):
    def __init__(self):
        pass

    def evaluate(self, indiv_def, indiv, training_datapair, validation_datapair=None)
        for block_index, block in enumerate(indiv.blocks):
            if block.need_evaluate:
                training_datapair = indiv_def[block_index].evaluate(block, training_datapair, validation_datapair)

        indiv.output = training_datapair #TODO figure this out



class BlockStandardEvaluate(EvaluateDefinition):

    def evaluate(self, block_def, block, training_datapair, validation_datapair=None):
        self.reset_evaluation(block)

        # go solve

        output = []
        for output_index in range(block_def.main_count, block_def.main_count+block_def.output_count):
            output.append(block.evaluated[output_index])
        return output

    def reset_evaluation(self, block):
        block.evaluated = [None] * len(block.genome)
        block.output = None
        block.dead = False



class BlockMPIStandardEvaluate(EvaluateDefinition):

    def evaluate(self, block, training_datapair, validation_datapair=None):
        pass


class BlockPreprocessEvaluate(EvaluateDefinition):

    def evaluate(self, block, training_datapair, validation_datapair=None):
        pass


class BlockTensorFlowEvaluate(EvaluateDefinition):

    def evaluate(self, block, training_datapair, validation_datapair):
        pass
