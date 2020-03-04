'''
The genetic material of each individual will vary but the structural components will be the same.
This structure is defined in layers:

      STRUCTURE      |||||      DEFINED BY
individual structure |||||  defined by list of blocks
block structure      |||||  defined by shape/meta data, mate methods,
                     |||||    mutate methods, evaluate method, operators
                     |||||    or primitives, argument datatypes
'''

# packages
import sys
import os
from typing import List
from numpy import random as rnd
import numpy as np
from copy import deepcopy

# scripts
from shape import ShapeMetaDefinition
from mutate import MutateDefinition
from mate import MateDefinition
from evaluate import EvaluateDefinition
from operators import OperatorDefinition
from arguments import ArgumentDefinition


class BlockDefinition():
    def __init__(self,
                nickname: str,
                meta_def: ShapeMetaDefinition,
                operator_def: OperatorDefinition,
                argument_def: ArgumentDefinition,
                evaluate_def: EvaluateDefinition,
                mutate_def: MutateDefinition,
                mate_def: MateDefinition):

        
        # Meta:
        self.nickname = nickname
        self.meta_def = meta_def()
        for name, val in self.meta_def.__dict__.items():
            # quick way to take all attributes and add to self
            self.__dict__[name] = val
        # Mutate:
        self.mutate_def = mutate_def()
        self.prob_mutate = self.mutate_def.prob_mutate
        self.num_mutants = self.mutate_def.num_mutants
        # Mate:
        self.mate_def = mate_def()
        self.prob_mate = self.mate_def.prob_mate
        # Evaluate:
        self.evaluate_def = evaluate_def()
        # Operator:
        self.operator_def = operator_def()
        self.operator_dict = self.operator_def.operator_dict
        self.operator_dict["input"] = self.meta_def.input_dtypes
        self.operator_dict["output"] = self.meta_def.output_dtypes
        self.operators = self.operator_def.operators
        self.operator_weights = self.operator_def.weights
        # Argument:
        self.argument_def = argument_def()
        self.arg_count = self.argument_def.arg_count
        self.arg_types = self.argument_def.arg_types


    def init_block(self, block):
        '''
        define:
         * block.genome
         * block.args
         * block.need_evaluate
        '''
        block.need_evaluate = True
        self.fill_args(block)
        self.fill_genome(block)
        self.get_actives(block)

    def get_node_dtype(self, block, node_index: int, key: str):
        '''
        key returns that key-value from the respective node_dictionary
         * "inputs"
         * "args"
         * "output"
        '''
        if node_index < 0:
            # input_node
            return self.input_dtypes[-1*node_index-1]
        elif node_index >= self.main_count:
            # output_node
            return self.output_dtypes[node_index-self.main_count]
        else:
            # main_node
            node_ftn = block[node_index]["ftn"]
            oper_dict_value = self.operator_dict[node_ftn]
            return oper_dict_value[key]

    def get_random_input(self, block, req_dtype, min_=None, max_=None, exclude = []):
        '''
        note max_ is exclusive so [min_,max_)

        return None if we failed to find good input
        '''
        if min_ is None:
            min_ = -1*self.input_count
        if max_ is None:
            max_ = self.main_count
        
        choices = np.arange(min_, max_)
        for val in exclude:
            choices = np.delete(choices, np.where(choices==val))

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

    def get_random_ftn(self, req_dtype=None, exclude=[], return_all=False):
        '''
        words
        '''
        choices = np.array(self.operators)
        weights = np.array(self.operator_weights)
        
        for val in exclude:
            #note: have to delete from weights first because we use choices to get right index
            weights = np.delete(weights, np.where(choices==val))
            choices = np.delete(choices, np.where(choices==val))
        
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
            weights *= 1/weights.sum()

        if return_all:
            return rnd.choice(choices, size=len(choices), replace=False, p=weights)
        else:
            return rnd.choice(choices, p=weights)

    def get_random_arg(self, req_dtype, exclude=[]):
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

    def fill_args(self, block):
        block.args = [None]*self.arg_count
        for arg_index, arg_type in enumerate(self.arg_types):
            block.args[arg_index] = arg_type()

    def fill_genome(self, block):
        block.genome = [None]*self.genome_count
        block.genome[(-1*self.input_count):] = ["InputPlaceholder"]*self.input_count

        # fill main nodes
        for node_index in range(self.main_count):
            ftns = self.get_random_ftn(return_all=True)
            for ftn in ftns:
                # find inputs
                input_dtypes = self.operator_dict[ftn]["inputs"]
                input_index = [None]*len(input_dtypes)
                for ith_input, input_dtype in enumerate(input_dtypes):
                    input_index[ith_input] = self.get_random_input(block, req_dtype=input_dtype, max_=node_index)
                if None in input_index:
                    # failed to fill it in; try another ftn
                    continue
                else:
                    pass

                # find args
                arg_dtypes = self.operator_dict[ftn]["args"]
                arg_index = [None]*len(arg_dtypes)
                for ith_arg, arg_dtype in enumerate(arg_dtypes):
                    poss_arg_index = self.get_random_arg(req_dtype=arg_dtype)
                if None in arg_index:
                    # failed to fill it in; try another ftn
                    continue
                else:
                    pass

                # all complete
                block[node_index] = {"ftn": ftn,
                                    "inputs": input_index,
                                    "args": arg_index}
                break
            # error check that node got filled
            if block[node_index] is None:
                print("GENOME ERROR: no primitive was able to fit into current genome arrangment")
                exit()

        # fill output nodes
        for ith_output, node_index in enumerate(range(self.main_count, self.main_count+self.output_count)):
            req_dtype = self.output_dtypes[ith_output]
            block[node_index] = self.get_random_input(block, req_dtype=req_dtype)

    def get_actives(self, block):
        block.active_nodes = set(np.arange(self.main_count, self.main_count+self.output_count))
        block.active_args = set()
        #block.active_ftns = set()

        # add feeds into the output_nodes
        for node_input in range(self.main_count, self.main_count+self.output_count):
            block.active_nodes.update([block[node_input]])

        for node_index in reversed(range(self.main_count)):
            if node_index in block.active_nodes:
                # then add the input nodes to active list
                block.active_nodes.update(block[node_index]["inputs"])
                block.active_args.update(block[node_index]["args"])
                '''# if we need to check for learners...
                if (not block.has_learner) and ("learner" in block[node_index]["ftn"].__name__):
                    block.has_learner = True
                else:
                    pass'''
            else:
                pass

        # sort
        block.active_nodes = sorted(list(block.active_nodes))
        block.active_args = sorted(list(block.active_args))


    def mutate(self, indiv, block_index: int):
        self.mutate_def.mutate(indiv, block_index, self)
        self.get_actives(indiv[block_index])

    def mate(self, parent1, parent2, block_index: int):
        #children: List() = self.mate_def.mate(parent1, parent2, block_index)
        children = self.mate_def.mate(parent1, parent2, block_index)
        for child in children:
            self.get_actives(child[block_index])
        return children

    def evaluate(self, block_def, block, training_datapair, validation_datapair=None):
        # verify that the input data matches the expected datatypes
        # TODO make a rule that training_datapair always has to be a list??? would be easiest for code
        for input_dtype, input_data in zip(block_def.input_dtypes, training_datapair):
            if input_dtype != type(input_data):
                print("ERROR: datatypes don't match", type(input_data), input_dtype) # add a proper message here
                return

        output = self.evaluate_def.evaluate(block_def, block, training_datapair, validation_datapair)
        return output



class IndividualDefinition():
    def __init__(self,
                block_defs: List[BlockDefinition],
                mutate_def: MutateDefinition,
                mate_def: MateDefinition,
                evaluate_def: EvaluateDefinition):
        self.block_defs = block_defs
        self.block_count = len(block_defs)
        self.mutate_def = mutate_def()
        self.mate_def = mate_def()
        self.evaluate_def = evaluate_def()

    def __getitem__(self, block_index: int):
        return self.block_defs[block_index]


    def get_actives(self, indiv):
        for block_index, block in enumerate(indiv.blocks):
            self[block_index].get_actives(indiv[block_index])

    def mutate(self, indiv):
        mutants = self.mutate_def.mutate(self, indiv)
        return mutants

    def mate(self, parent1, parent2):
        children = self.mate_def.mate(self, parent1, parent2)
        return children

    def evaluate(self, indiv, training_datapair, validation_datapair=None):
        self.evaluate_def.evaluate(self, indiv, training_datapair, validation_datapair)