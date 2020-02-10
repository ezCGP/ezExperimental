# packages
from typing import List

# scripts





class Block_Material():
	def __init__(self, skeleton: Block_Definition):
		Block_Declaration.init_block(self, skeleton) #create shape of genome and args
		Block_Declaration.fill_args(self, skeleton)



class Individual_Material():
	'''
	lightweight class to hold unique genetic material for each individual
	should only have a few attributes and getitem d-unders
	all attributes except score, should be a list of len(blocks)
	'''

	def __init__(self, block_defs: List(Block_Definition)):
		self.blocks = []
		for block_def in block_defs:
			self.blocks.append( Block_Material(block_def) )












		self.args = [None]*skeleton.block_count
		self.genome = [None]*skeleton.block_count
		for block_i in skeleton.block_count:
			self.genome[block_i] = Block_Material(skeleton.blocks[i])


			self.args[i] = struct[i].fillArgs() #something like that
		Individual_Structure.evaluate(self)
		self.score = 0 #?

	def __setitem__(self, block_node_index: tuple, value):
		block_index, node_index = block_node_index
		self.genome[block_index][node_index] = value

	def __getitem__(self, block_i: int):
		_gen = self.genome[block_i]
		_arg = self.args[block_i]
		_eval = self.need_evaluate[block_i]
		return _gen, _arg, _eval
