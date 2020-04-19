'''
Class which contains the generic evaluateDefinition interface as well as graphEvaluateDefinitions which use
the interface.

    evaluateDefinition is based upon the idea that every object is evaluable
    graphEvaluateDefinition is based upon the idea that there are three steps in graph creation:
        build_graph,
        run_graph,
        reset_graph

'''

# packages
import sys
import os
from abc import ABC, abstractmethod
import traceback
import tensorflow as tf
from ezData.data_pair import DataPair
from ezData.dataset import DataSet
from sklearn.metrics import accuracy_score as accuracy
import numpy as np
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
    @abstractmethod
    def evaluate(self):
        pass

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
    def build_graph(self):  # fill Genome?
        """
        This will fill the genome and the arguments. Randomize connections.
        Original code is in genome.py. FillArgs and
        """
        pass

    @abstractmethod
    def reset_graph(self):  # resetAttrVals
        """
        Original code in genome.py's resetEvalAttr function.
        """
        pass

    def run_graph(self):
        pass



# class IndividualStandardEvaluate(EvaluateDefinition):
#     def __init__(self):
#         pass
#
#     def evaluate(self, indiv_def, indiv, data):
#         for block_index, block in enumerate(indiv.blocks):
#             block_def = indiv_def[block_index]
#             if block.need_evaluate:
#                 training_datapair = indiv_def[block_index].evaluate(block_def, block, data)
#
#         indiv.output = training_datapair #TODO figure this out
#
#     def reset_evaluation(self):
#         pass


class IndividualStandardEvaluate(EvaluateDefinition):
    def __init__(self):
        pass

    def evaluate(self, indiv_def, indiv, data):
        for block_index, block in enumerate(indiv.blocks):
            block_def = indiv_def[block_index]
            if block.need_evaluate:
                try:
                    data = indiv_def[block_index].evaluate(block_def, block, data)
                    block.evaluated = [None] * len(block.genome)  # reset
                except Exception as e:
                    print("individual failed to evaluate at block_index", block_index)
                    print(e)
                    block.evaluated = [None] * len(block.genome)
                    traceback.print_exc()
                    indiv.output = None
                    return
        indiv.output = data #TODO figure this out

    def reset_evaluation(self):
        pass



class BlockStandardEvaluate(EvaluateDefinition):

    def evaluate(self, block_def, block, data_pair: DataPair):
        training_datapair = data_pair.get_data()[0]
        self.reset_evaluation(block)

        # add input data
        for i, data_input in enumerate(training_datapair):
            block.evaluated[-1*(i+1)] = data_input

        # go solve
        for node_index in block.active_nodes:
            if node_index < 0:
                # do nothing. at input node
                continue
            elif node_index >= block_def.main_count:
                # do nothing NOW. at output node. we'll come back to grab output after this loop
                continue
            else:
                # main node. this is where we evaluate
                function = block[node_index]["ftn"]
                
                inputs = []
                node_input_indices = block[node_index]["inputs"]
                for node_input_index in node_input_indices:
                    inputs.append(block.evaluated[node_input_index])

                args = []
                node_arg_indices = block[node_index]["args"]
                for node_arg_index in node_arg_indices:
                    args.append(block.args[node_arg_index].value)

                #print(function, inputs, args)
                block.evaluated[node_index] = function(*inputs, *args)
                '''try:
                    self.evaluated[node_index] = function(*inputs, *args)
                except Exception as e:
                    print(e)
                    self.dead = True
                    break'''

        output = []
        for output_index in range(block_def.main_count, block_def.main_count+block_def.output_count):
            output.append(block.evaluated[block.genome[output_index]])
        return output

    def reset_evaluation(self, block):
        block.evaluated = [None] * len(block.genome)
        block.output = None
        block.dead = False


class BlockMPIStandardEvaluate(EvaluateDefinition):

    def evaluate(self, block, training_datapair, validation_datapair=None):
        pass


class BlockAugmentationEvaluate(BlockStandardEvaluate):

    def evaluate(self, block_def, block, dataset: DataSet):

        self.reset_evaluation(block)  # TODO most of this code can be abstracted out as a global to all blocks

        # add input data
        block.evaluated[-1] = dataset.augmentation_pipeline  # assign input to last index

        # go solve
        for node_index in block.active_nodes:
            if node_index < 0:
                # do nothing. at input node
                continue
            elif node_index >= block_def.main_count:
                # do nothing NOW. at output node. we'll come back to grab output after this loop
                continue
            else:
                # main node. this is where we evaluate
                function = block[node_index]["ftn"]

                inputs = []
                node_input_indices = block[node_index]["inputs"]
                for node_input_index in node_input_indices:
                    inputs.append(block.evaluated[node_input_index])

                args = []
                node_arg_indices = block[node_index]["args"]
                for node_arg_index in node_arg_indices:
                    args.append(block.args[node_arg_index].value)

                # print(function, inputs, args)
                block.evaluated[node_index] = function(*inputs, *args)
                '''try:
                    self.evaluated[node_index] = function(*inputs, *args)
                except Exception as e:
                    print(e)
                    self.dead = True
                    break'''

        output = block.evaluated[block.genome[block_def.main_count]]  # modified augmentor
        dataset.augmentation_pipeline = output
        tf.keras.backend.clear_session()
        return dataset


class BlockPreprocessEvaluate(BlockStandardEvaluate):

    def evaluate(self, block_def, block, dataset: DataSet):
        self.reset_evaluation(block)  # TODO most of this code can be abstracted out as a global to all blocks

        # add input data
        block.evaluated[-1] = dataset.preprocess_pipeline

        # go solve
        for node_index in block.active_nodes:
            if node_index < 0:
                # do nothing. at input node
                continue
            elif node_index >= block_def.main_count:
                # do nothing NOW. at output node. we'll come back to grab output after this loop
                continue
            else:
                # main node. this is where we evaluate
                function = block[node_index]["ftn"]

                inputs = []
                node_input_indices = block[node_index]["inputs"]
                for node_input_index in node_input_indices:
                    inputs.append(block.evaluated[node_input_index])

                args = []
                node_arg_indices = block[node_index]["args"]
                for node_arg_index in node_arg_indices:
                    args.append(block.args[node_arg_index].value)

                # print(function, inputs, args)
                block.evaluated[node_index] = function(*inputs, *args)
                '''try:
                    self.evaluated[node_index] = function(*inputs, *args)
                except Exception as e:
                    print(e)
                    self.dead = True
                    break'''

        output = block.evaluated[block.genome[block_def.main_count]]  # modified augmentor
        dataset.preprocess_pipeline = output
        tf.keras.backend.clear_session()
        return dataset


class BlockTensorFlowEvaluate(BlockStandardEvaluate):
    """main -> universe -> individual evaluate -> tensorblock evaluate"""
    def evaluate(self, block_def, block, dataset: DataSet):
        """
        block_def : specifies specific params of the block.
        block: contains graph structure
        dataset: contains training dataset.

        returns: the predicted labels of the the validation set contained in dataset
        """

        gpus = tf.config.experimental.list_physical_devices('GPU')
      #  tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 3)])

       # with tf.device(':'.join(gpus[0].name.split(':')[1:])):
        if 1 == 1:
            self.reset_evaluation(block)  # TODO most of this code can be abstracted out as a global to all blocks
            num_classes = 10

            # add input data
            sample, _ = dataset.next_batch_train(1)

            inputshape = sample[0].shape
            input_ = tf.keras.layers.Input(inputshape)  # dataset.augmentor_pipeline
            block.evaluated[-1] = input_

            # go solve
            for node_index in block.active_nodes:
                if node_index < 0:
                    # do nothing. at input node
                    continue
                elif node_index >= block_def.main_count:
                    # do nothing NOW. at output node. we'll come back to grab output after this loop
                    continue
                else:
                    # main node. this is where we evaluate
                    function = block[node_index]["ftn"]

                    inputs = []
                    node_input_indices = block[node_index]["inputs"]
                    for node_input_index in node_input_indices:
                        inputs.append(block.evaluated[node_input_index])

                    args = []
                    node_arg_indices = block[node_index]["args"]
                    for node_arg_index in node_arg_indices:
                        args.append(block.args[node_arg_index].value)

                    # print(function, inputs, args)
                    block.evaluated[node_index] = function(*inputs, *args)
                    '''try:
                        self.evaluated[node_index] = function(*inputs, *args)
                    except Exception as e:
                        print(e)
                        self.dead = True
                        break'''

            output = block.evaluated[block.genome[block_def.main_count]]  # modified augmentor

            #  flatten the output node and perform a softmax
            flat_out = tf.keras.layers.Flatten()(output)
            logits = tf.keras.layers.Dense(num_classes)(flat_out)
            softmax = tf.keras.activations.softmax(logits, axis= 1)

            #  construct model from "dummy" input and softmax output
            model = tf.keras.Model(input_, softmax, name="dummy")
            opt = tf.keras.optimizers.Adam(learning_rate=0.001)
            #init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
            model.compile(loss = "categorical_crossentropy", optimizer = opt)

            #  TODO should extract training parameters from dataset object
            batch_size = 256
            n_epochs = 15  # TODO set variable n_epochs changeable from problem
            mini_batches = len(dataset.x_train) // batch_size

            # validation params for early stopping
            x_val_norm, _ = dataset.preprocess_test_data()
            valCheckTime = 5  # every 5 epochs get validation accuracy
            old_validation_accuracy = 0
            predictions = None # stores validation predictions
            for i in range(n_epochs):
                training_loss = 0

                for b in range(mini_batches):
                    batchX, batchY = dataset.next_batch_train(batch_size)
                    training_loss+= model.train_on_batch(batchX , batchY)

                if i % valCheckTime == 0:  # early stopping
                        predictions = model.predict(x_val_norm)

                        validation_accuracy =  accuracy(np.argmax(dataset.y_test, axis = 1), 
                            np.argmax(predictions, axis=1))

                        if validation_accuracy < old_validation_accuracy:
                            print("old Val Accuracy", old_validation_accuracy, "was better than current val accuracy", validation_accuracy)
                            tf.keras.backend.clear_session()
                            return predictions
                        print("validation accuracy", validation_accuracy)
                        old_validation_accuracy = validation_accuracy
                print("Epoch", i, "training loss", training_loss)

            tf.keras.backend.clear_session()
            return model.predict(x_val_norm)

        #dataset -> augmentationBlock-> return dataset -> preprocessingBlock -> return dataset -> BlockTensorFlowEvaluate -> return outputs of val set
