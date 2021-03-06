# operators/primitives mostly for symbolic regression
import numpy as np
operDict = {}


def add_ff2f(a,b):
    return np.add(a,b)
operDict[add_ff2f] = {"inputs": [np.float64, np.float64],
                      "output": np.float64,
                      "args": []
                    }

def add_fa2a(a,b):
    return np.add(a,b)
operDict[add_fa2a] = {"inputs": [np.ndarray, np.float64],
                      "output": np.ndarray,
                      "args": []
                      }
"""
operDict[add_fa2a] = {"inputs": [np.ndarray],
                      "output": np.ndarray,
                      "args": [FloatSmall],
                      "num_args": 3}
"""

def add_aa2a(a,b):
    return np.add(a,b)
operDict[add_aa2a] = {"inputs": [np.ndarray, np.ndarray],
                      "output": np.ndarray,
                      "args": []
                      }


def sub_ff2f(a,b):
    return np.subtract(a,b)
operDict[sub_ff2f] = {"inputs": [np.float64, np.float64],
                      "output": np.float64,
                      "args": []
                      }

def sub_fa2a(a,b):
    return np.subtract(a,b)
operDict[sub_fa2a] = {"inputs": [np.float64, np.ndarray],
                      "output": np.ndarray,
                      "args": []
                      }
#operDict[sub_fa2a] = {"inputs": [np.ndarray],
#                     "output": np.ndarray,
#                     "args": [FloatSmall],
#                     "num_args": 3}

def sub_aa2a(a,b):
    return np.subtract(a,b)
operDict[sub_aa2a] = {"inputs": [np.ndarray, np.ndarray],
                      "output": np.ndarray,
                      "args": []
                      }
#operDict[sub_aa2a] = {"inputs": [np.ndarray, np.ndarray],
#                     "output": np.ndarray,
#                     "args": [],
#                     "num_args": 3}

def mul_ff2f(a,b):
    return np.multiply(a,b)
operDict[mul_ff2f] = {"inputs": [np.float64, np.float64],
                      "output": np.float64,
                      "args": []
                      }

def mul_fa2a(a,b):
    return np.multiply(a,b)
operDict[mul_fa2a] = {"inputs": [np.float64, np.ndarray],
                      "output": np.ndarray,
                      "args": []
                      }
#operDict[mul_fa2a] = {"inputs": [np.ndarray],
#                     "output": np.ndarray,
#                     "args": [FloatSmall],
#                     "num_args": 3}

def mul_aa2a(a,b):
    return np.multiply(a,b)
operDict[mul_aa2a] = {"inputs": [np.ndarray, np.ndarray],
                      "output": np.ndarray,
                      "args": []
                      }
#operDict[mul_aa2a] = {"inputs": [np.ndarray, np.ndarray],
#                     "output": np.ndarray,
#                     "args": [],
#                     "num_args": 3}