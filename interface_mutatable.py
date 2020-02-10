# packages
from abc import ABC, abstractmethod
import pickle as pkl

#scripts


class Mutatable(ABC):

	@abstractmethod
	def mutate():
		pass


	def hash(self):
		return "FILL_IN_LATER"

	def save(self):
		'''
		the intention here is for easy 'portability'...
		have the option to read in a pkl'ed mutable class instead of reprogramming it
		'''
		with open(self.hash(), 'wb') as f:
			pkl.dump(self, f)