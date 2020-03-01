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


"""
TensforFlowGraph Evaluate vs. PyTorch vs. Keras Evaluation
"""


class GraphEvaluateDefinition(EvaluateDefinition):
    '''
    attempt at abstracting what an EvaluateDefinition will look like for a 
    computational graph block like tensorflow, pytorch, or keras

    these are just ideas
    
    Edit notes (Sam): TF 2.0 has a tf.function class that builds computational graphs automatically (is recommended), see operators.py
    '''

    def evaluate(self, block, training_datapair, validation_datapair=None):
        self.build_graph()
        self.train_graph()
        return self.run_graph()

    """
    Rodd_layout --conversion steps--> Graph Layout
    Graph:
        Nodes:
            Inputs: [Input Nodes]
            Args: [Arg Types]
            Function: [Dense, ResNet]
        Edges:
    """

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


class TfGraphEvaluateDefinition(EvaluateDefinition):
    '''
    attempt at abstracting what an EvaluateDefinition will look like for a
    computational graph block like tensorflow, pytorch, or keras

    these are just ideas

    Edit notes (Sam): TF 2.0 has a tf.function class that builds computational graphs automatically (is recommended), see operators.py
    '''

    def __init__(self, operator_dict, genome_main_count, genome_output_dtypes, operators, operator_weights):
        super().__init__()
        self.operators, self.operator_weights = operators, operator_weights
        self.operator_dict, self.genome_main_count, self.genome_output_dtypes = operator_dict, genome_main_count, genome_output_dtypes

    def random_ftn(self, req_dtype=None, exclude=[], return_all=False):
        '''
        words
        '''
        choices = np.array(self.operators)
        weights = np.array(self.operator_weights)

        for val in exclude:
            # note: have to delete from weights first because we use choices to get right index
            weights = np.delete(weights, np.where(choices == val))
            choices = np.delete(choices, np.where(choices == val))

        # now check the output dtypes match
        if req_dtype is not None:
            delete = []
            for ith_choice, choice in enumerate(choices):
                if self.operator_dict[choice]["output"] != req_dtype:
                    delete.append(ith_choice)
            weights = np.delete(weights, delete)
            choices = np.delete(choices, delete)

        if weights.sum() < 1 - 1e-3:
            # we must have removed some values...normalize
            weights *= 1 / weights.sum()

        if return_all:
            return rnd.choice(choices, size=len(choices), replace=False, p=weights)
        else:
            return rnd.choice(choices, p=weights)

    def get_node_dtype(self, block, node_index: int, key: str):
        '''
        key returns that key-value from the respective node_dictionary
         * "inputs"
         * "args"
         * "output"
        '''
        if node_index < 0:
            # input_node
            return self.input_dtypes[-1 * node_index - 1]
        elif node_index >= self.main_count:
            # output_node
            return self.output_dtypes[node_index - self.main_count]
        else:
            # main_node
            node_ftn = block[node_index]["ftn"]
            oper_dict_value = self.operator_dict[node_ftn]
            return oper_dict_value[key]

    def random_input(self, block, req_dtype, min_=None, max_=None, exclude=[]):
        '''
        note max_ is exclusive so [min_,max_)

        return None if we failed to find good input
        '''
        if min_ is None:
            min_ = -1 * self.input_count
        if max_ is None:
            max_ = self.main_count

        choices = np.arange(min_, max_)
        for val in exclude:
            choices = np.delete(choices, np.where(choices == val))

        if len(choices) == 0:
            # nothing left to choose from
            return None
        else:
            '''
            exhuastively try each choice to see if we can get datatypes to match
            '''
            poss_inputs = np.random.choice(a=choices, size=len(choices), replace=False)
            for input_index in poss_inputs:
                input_dtype = self.get_node_dtype(block, input_index, "output")
                if req_dtype == input_dtype:
                    return input_index
                else:
                    pass
            # none of the poss_inputs worked, failed to find matching input
            return None

    def random_arg(self, req_dtype, exclude=[]):
        '''
        words
        '''
        choices = []
        for arg_index, arg_type in enumerate(self.arg_types):
            if (arg_type == req_dtype) and (arg_index not in exclude):
                choices.append(arg_index)

        if len(choices) == 0:
            return None
        else:
            return rnd.choice(choices)

    def build_graph(self):
        genome = []
        # Genome - Main Nodes
        for node_index in range(self.genome_main_count):
            # randomly select all function, find a previous node with matching output_dtype matching our function input_dtype
            # iterate through each function until you find matching output and input data types
            ftns = self.random_ftn()
            for ftn in ftns:
                # connect to a previous node
                input_dtypes = self.operator_dict[ftn]["inputs"]
                input_nodes = []  # [None]*len(input_dtypes)
                for input_dtype in input_dtypes:
                    input_nodes.append(self.random_input(max_=node_index, req_dtype=input_dtype))
                if None in input_nodes:
                    # couldn't find a matching input + output value. try the next ftn
                    break
                else:
                    # go on to find args
                    pass
                # connect to any required arguments
                arg_dtypes = self.operator_dict[ftn]["args"]
                arg_nodes = []
                for arg_dtype in arg_dtypes:
                    arg_nodes.append(self.random_arg(req_dtype=arg_dtype))
                # assign all values to the genome
                genome[node_index] = {"ftn": ftn,
                                      "inputs": input_nodes,
                                      "args": arg_nodes}
                # we found a ftn that works, move on to next node_index
                break
        # Genome - Output Nodes
        for i, output_dtype in enumerate(self.genome_output_dtypes):
            genome[self.genome_main_count + i] = self.random_input(min_=0, max_=self.genome_main_count,
                                                                   dtype=output_dtype)

    """
    How does this work
    """

    def reset_graph(self):
        pass

    """
    How does this work
    """

    def train_graph(self):
        pass

    """
    How does this work
    """

    def run_graph(self):
        pass


class IndividualStandardEvaluate(EvaluateDefinition):
    def __init__(self):
        pass

    def evaluate(self, indiv_def, indiv, training_datapair, validation_datapair=None)
        for block_index, block in enumerate(indiv.blocks):
            if block.need_evaluate:
                training_datapair = indiv_def[block_index].evaluate(block, training_datapair, validation_datapair)

        indiv.output = training_datapair  # TODO figure this out


class BlockStandardEvaluate(EvaluateDefinition):

    def evaluate(self, block_def, block, training_datapair, validation_datapair=None):
        self.reset_evaluation(block)

        # go solve

        output = []
        for output_index in range(block_def.main_count, block_def.main_count + block_def.output_count):
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
