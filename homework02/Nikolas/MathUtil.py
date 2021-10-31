import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoidprime(x):
    sig = sigmoid(x)
    return sig*(1.0-sig)