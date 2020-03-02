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

        # Genome - Argument List
        self.args_count = genome_arg_count
        self.args = [None]*genome_arg_count
        self.arg_methods = [None]
        self.arg_weights = [None]

        # Genome - Ftn List
        self.genome = [None]*(len(genome_input_dtypes)+genome_main_count+len(genome_output_dtypes))
        self.genome_count = len(self.genome)

        # Block - Genome List - Input Nodes
        self.genome_input_dtypes = genome_input_dtypes # a list of data types. the exact values will be assigned at evaluation step
        self.genome_input_count = len(genome_input_dtypes)
        self.genome[-1*len(genome_input_dtypes):] = ["InputPlaceholder"]*len(genome_input_dtypes)#block_inputs

        # Block - Genome List - Main Nodes
        self.genome_main_count = genome_main_count
        self.ftn_methods = [None]
        self.ftn_weights = [None]

        # Block - Genome List - Outputs Nodes
        self.genome_output_dtypes = genome_output_dtypes # a list of data types. the exact values will be evaluated at evaluation step
        self.genome_output_count = len(genome_output_dtypes)


    def get_node_type(self, node_index, arg_dtype=False, input_dtype=False, output_dtype=False):
        if node_index < 0:
            # then it's a Block Input Node
            return self.genome_input_dtypes[-1*node_index-1] # -1-->0, -2-->1, -3-->2
        elif node_index >= self.genome_main_count:
            # then it's a Block Output Node
            return self.genome_output_dtypes[node_index-self.genome_main_count]
        else:
            # then it's a Block Main Node
            pass
        try:
            ftn = self[node_index]["ftn"]
        except TypeError:
            print('Input/output dtype is incompatible with operator functions')
            quit()
        ftn_dict = self.operator_dict[ftn]
        if input_dtype:
            # get the required input data types for this function
            return ftn_dict["inputs"] # will return a list
        elif output_dtype:
            # get the output data types for this function
            return ftn_dict["outputs"] # returns a single value, not a list...arity is always 1 for outputs
        elif arg_dtype:
            return ftn_dict["args"]
        else:
            print("ProgrammerError: script writer didn't assign input/output dtype for getNodeType method")
            exit()


    def random_ftn(self, only_one=False, exclude=None, output_dtype=None):
        choices = self.ftn_methods
        weights = self.ftn_weights
        if exclude is not None:
            delete = []
            for val in exclude:
                # value == list doesn't work when the value is a function
                for c, choice in enumerate(choices):
                    if val==choice:
                        delete.append(c)

            if len(choices) != 1:
                #print("Should not call mutate since only one function available. Returning original function")

                choices = np.delete(choices, delete)
                weights = np.delete(weights, delete)
                weights /= weights.sum() # normalize weights
        else:
            pass
        if output_dtype is not None:
            # force the selected function to have the output_dtype
            delete = []
            for c, choice in enumerate(choices):
                if self.operator_dict[choice]["outputs"]!=output_dtype:
                    delete.append(c)
            choices = np.delete(choices, delete)
            weights = np.delete(weights, delete)
            weights /= weights.sum() # normalize weights
        if only_one:
            return np.random.choice(a=choices, size=1, p=weights)[0]
        else:
            return np.random.choice(a=choices, size=len(self.ftn_methods), replace=False, p=weights)


    def random_input(self, dtype, min_='default', max_='default', exclude=None):
        # max_ is one above the largest integer to be drawn
        # so if we are randomly finding nodes prior to a given node index, set max_ to that index
        if min_=='default':
            min_=-1*self.genome_input_count
        if max_=='default':
            max_=self.genome_main_count
        choices = np.arange(min_, max_)
        if exclude is not None:
            for val in exclude:
                choices = np.delete(choices, np.where(choices==val))
        else:
            pass
        if len(choices) == 0:
            #nothing to choices from...very rare but very possible
            return None
        possible_nodes = np.random.choice(a=choices, size=len(choices), replace=False)
        # iterate through each input until we find a datatype that matches dtype
        for poss_node in possible_nodes:
            poss_node_outputdtype = self.get_node_type(node_index=poss_node, output_dtype=True)
            # note, we currently only allow for arity of 1 for our ftns...so getNodeType will return a value not a list if output_dtype is True
            if dtype == poss_node_outputdtype:
                return poss_node
            else:
                pass
        # if we got this far then we didn't find a match
        return None


    def random_arg(self, dtype, exclude=None):
        # have to assume here that each arg_dtype will have at least two in each self.args...or else when we mutate() we would get same arg
        choices = []
        for arg_index in range(self.args_count):
            arg_dtype = type(self.args[arg_index]).__name__
            if arg_dtype == dtype:
                if (exclude is not None) and (arg_index in exclude):
                    continue
                choices.append(arg_index)
        if len(choices) == 0:
            print("UserInputError: A ftn was provided without having its required (or invalid) data type in the arguments")
            exit()
        else:
            return np.random.choice(a=choices, size=1)[0]


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
                    input_nodes.append(self.random_input(max_=node_index, dtype=input_dtype))
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
                    arg_nodes.append(self.random_arg(dtype=arg_dtype))
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
