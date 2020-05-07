#import individual.individual_definition
#from .individual import individual_definition

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#import v04.individual.individual_definition
from v04.individual import individual_definition
print(individual_definition.ting)


print(__file__)
print(os.path.realpath(__file__))
print(os.path.dirname(os.path.realpath(__file__)))
print(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))