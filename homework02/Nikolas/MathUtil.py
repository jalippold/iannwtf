import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidprime(x):
    sig = sigmoid(x)
    return sig*(1-sig)