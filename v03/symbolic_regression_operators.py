# operators/primitives mostly for symbolic regression
import numpy as np
operDict = {}

'''
Each of these functions are operators for the basic "symbolic_regression_operators" module used in Symbolic Regression.

The operDict that defines the metadata for each of these primitives are outlined as follows:

    inputs: type of the arguments
    output: type of the output
    weight: likelihood operator is chosen when initializing genome
'''

def add_ff2f(a,b):
    return np.add(a,b)

operDict[add_ff2f] = {"inputs": [np.float64, np.float64],
                      "output": np.float64,
                      "args": [],
                      "weight": 1
                    }

def add_fa2a(a,b):
    return np.add(a,b)
operDict[add_fa2a] = {"inputs": [np.ndarray, np.float64],
                      "output": np.ndarray,
                      "args": [],
                      "weight": 1
                      }

def add_aa2a(a,b):
    return np.add(a,b)
operDict[add_aa2a] = {"inputs": [np.ndarray, np.ndarray],
                      "output": np.ndarray,
                      "args": [],
                      "weight": 1
                      }


def sub_ff2f(a,b):
    return np.subtract(a,b)
operDict[sub_ff2f] = {"inputs": [np.float64, np.float64],
                      "output": np.float64,
                      "args": [],
                      "weight": 1
                      }

def sub_fa2a(a,b):
    return np.subtract(a,b)
operDict[sub_fa2a] = {"inputs": [np.float64, np.ndarray],
                      "output": np.ndarray,
                      "args": [],
                      "weight": 1
                      }

def sub_aa2a(a,b):
    return np.subtract(a,b)
operDict[sub_aa2a] = {"inputs": [np.ndarray, np.ndarray],
                      "output": np.ndarray,
                      "args": [],
                      "weight": 1
                      }

def mul_ff2f(a,b):
    return np.multiply(a,b)
operDict[mul_ff2f] = {"inputs": [np.float64, np.float64],
                      "output": np.float64,
                      "args": [],
                      "weight": 1
                      }

def mul_fa2a(a,b):
    return np.multiply(a,b)
operDict[mul_fa2a] = {"inputs": [np.float64, np.ndarray],
                      "output": np.ndarray,
                      "args": [],
                      "weight": 1
                      }

def mul_aa2a(a,b):
    return np.multiply(a,b)
operDict[mul_aa2a] = {"inputs": [np.ndarray, np.ndarray],
                      "output": np.ndarray,
                      "args": [],
                      "weight": 1
                      }

