'''
required inputs:
 * a single IndividualMaterial (already deepcopied if applicable)
 * a single int for which block index we should mutate

no outputs; the mutate method will alter the given individual. it is expected
that the original individual has already been deepcopied.
'''

# packages
import sys
import os
from numpy import random as rnd
import numpy as np
from copy import deepcopy

# scripts


def mutate_1(indiv, block_index: int):
    pass

def mutate_2(indiv, block_index: int):
    pass


def mutate_single_input(indiv, block_index: int, block_def):
    '''
    pick nodes at random and mutate the input-index until an active node is selected

    when mutating inputs, it will look for a node that outputs the matching datatype of the current node's input
    so it can fail at doing so and won't mutate that node
    '''
    block = indiv[block_index]
    choices = np.arange(block_def.main_count+block_def.output_count)
    choices = rnd.choice(choices, size=len(choices), replace=False) #randomly reorder
    for node_index in choices:
        if node_index < block_def.main_count:
            # then we are mutating a main-node (expect a node-dict)
            num_inputs_into_node = len(block[node_index]["inputs"])
            ith_input = rnd.choice(np.arange(num_inputs_into_node))
            current_input_index = block[node_index]["inputs"][ith_input]
            req_dtype = block_def.get_node_dtype(block, node_index, "inputs")[ith_input]
            new_input = block_def.get_random_input(block, req_dtype=req_dtype, max_=node_index, exclude=[current_input_index])
            if new_input is None:
                # failed to find new input, will have to try another node to mutate
                continue
            else:
                block[node_index]["inputs"][ith_input] = new_input
                if node_index in block.active_nodes:
                    # active_node finally mutated
                    break
                else:
                    pass
        else:
            # then we are mtuating an output-node (expect a int index value)
            current_output_index = block[node_index]
            req_dtype = block_def.output_dtypes[node_index-block_def.main_count]
            new_output_index = block_def.get_random_input(block, req_dtype=req_dtype, min_=0, exclude=[current_output_index])
            if new_output_index is None:
                # failed to find new node
                continue
            else:
                block[node_index] = new_output_index
                break # active_node finally mutated


def mutate_single_ftn(indiv, block_index: int, block_def):
    '''
    pick nodes at random and mutate the ftn-index until an active node is selected

    will mutate the function to anything with matching input/arg dtype.
    if the expected input datatypes don't match the current genome,
    it will find a new input/arg that will match
    '''
    block = indiv[block_index]
    choices = np.arange(block_def.main_count)
    choices = rnd.choice(choices, size=len(choices), replace=False) #randomly reorder
    for node_index in choices:
        # note, always will be a main_node
        current_ftn = block[node_index]["ftn"]
        req_output_dtype = block_def.operator_dict[current_ftn]["output"]
        new_ftn = block_def.get_random_ftn(req_dtype=req_output_dtype, exclude=[current_ftn])

        # make sure input_dtypes match
        req_input_dtypes = block_def.operator_dict[new_ftn]["inputs"]
        new_inputs = [None]*len(req_input_dtypes)
        for input_index in block[node_index]["inputs"]:
            exist_input_dtype = block_def.get_node_dtype(block, input_index, "output") #instead of verify from current node, goes to input
            for ith_input, (new_input, input_dtype) in enumerate(zip(new_inputs, req_input_dtypes)):
                if (new_input is None) and (input_dtype == exist_input_dtype):
                    # then we can match an existing input with one of our required inputs
                    new_inputs[ith_input] = input_index
                else:
                    pass
        # now try and fill in anything still None
        for ith_input, (new_input, input_dtype) in enumerate(zip(new_inputs, req_input_dtypes)):
            if new_input is None:
                new_inputs[ith_input] = block_def.get_random_input(block, req_dtype=input_dtype, max_=node_index)
        # if there is still 'None' then we failed to fit this ftn in...try another ftn
        if None in new_inputs:
            continue
        else:
            pass

        # make sure arg_dtypes match
        req_arg_dtypes = block_def.operator_dict[new_ftn]["args"]
        new_args = [None]*len(req_arg_dtypes)
        exist_arg_dtypes = block_def.get_node_dtype(block, node_index, "args")
        for arg_index, exist_arg_dtype in zip(block[node_index]["args"], exist_arg_dtypes):
            for ith_arg, (new_arg, req_arg_dtype) in enumerate(zip(new_args, req_arg_dtypes)):
                if (new_arg is None) and (req_arg_dtypes == exist_arg_dtype):
                    new_args[ith_arg] = arg_index
                else:
                    pass
        # now try and fill in anything still None
        for ith_arg, (new_arg, req_arg_dtype) in enumerate(zip(new_args, req_arg_dtypes)):
            if new_arg is None:
                new_args[ith_arg] = block_def.get_random_arg(req_dtype=req_arg_dtype)
        # if there is still 'None' then we failed to fit this ftn ...try another ftn
        if None in new_args:
            continue
        else:
            pass

        # at this point we found a ftn and fit inputs and args
        block[node_index]["ftn"] = new_ftn
        block[node_index]["inputs"] = new_inputs
        block[node_index]["args"] = new_args
        if node_index in block.active_nodes:
            # active_node finally mutated
            break
        else:
            pass


def mutate_single_argindex(indiv, block_index: int, block_def):
    '''
    words
    '''
    block = indiv[block_index]
    if len(block.active_args) > 0:
        # then there is something to mutate
        choices = [] # need to find those nodes with 'args' filled
        #weights = [] # option to sample node_index by the number of args for each node
        for node_index in range(block_def.main_count):
            if len(block[node_index]["args"]) > 0:
                choices.append(node_index)
                #weights.append(len(block[node_index]["args"]))
            else:
                pass

        choices = rnd.choice(choices, size=len(choices), replace=False) #randomly reorder
        for node_index in choices:
            ith_arg = rnd.choice(np.arange(len(block[node_index]["args"])))
            current_arg = block[node_index]["args"][ith_arg]
            arg_dtype = block_def.get_node_dtype(block, node_index, "args")[ith_arg]
            new_arg = block_def.get_random_arg(arg_dtype, exclude=[current_arg])
            if new_arg is None:
                # failed to find a new_arg
                continue
            else:
                block[node_index]["args"][ith_arg] = new_arg
                if node_index in block.active_nodes:
                    # active_node finally mutated
                    break
                else:
                    pass
    else:
        # won't actually mutate
        pass


def mutate_single_argvalue(indiv, block_index: int, block_def):
    '''
    words
    '''
    block = indiv[block_index]
    if len(block.active_args) > 0:
        # then there is something to mutate
        choices = np.arange(self.args_count)
        choices = rnd.choice(choices, size=len(choices), replace=False) #randomly reorder
        for arg_index in choices:
            block.args[arg_index].mutate()
            if arg_index in block.active_args:
                # active_arg finally mutated
                break
            else:
                pass
    else:
        # won't actually mutate
        pass

'''
# tried to tuck these all under a class so I can leverage the all() method
but if we don't make an instance of the class... ie mutmeths = MutateMethods()
then we can't use the 'self' value. and it seems slopier to try and make an
instance of this class so I gave up on it...instead just list the methods

class MutateMethods():

    def all(self):
        
        #gather all callable methods (excluding 'all'), and return
        
        methods = []
        for name, val in type(self).__dict__.items():
            if (callable(val)) and (not 'all'):
                methods.append(name)
        return methods

    def mutate_1(self, indiv: IndividualMaterial, block_index: int):
        pass

    def mutate_2(self, indiv: IndividualMaterial, block_index: int):
        pass
'''