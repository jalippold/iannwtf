import numpy as np
from MathUtil import *

class Perceptron():
    def __init__(self, input_units):
        """Perceptron with bias and i ingoing weighted activations"""
        self.bias = np.random.randn()
        self.weights = np.array([np.random.randn() for _ in range(input_units)])
        self.alpha = 1.0    # according to homework sheet this should be 1.0
        self.output = None
        self.inputs = None
        self.drive = None

    def forward_step(self, inputs):
        if len(inputs) is not len(self.weights):
            raise IndexError(f"Length of input ({len(inputs)}) does not match length of weights ({len(self.weights)})!")
        # remember input/activations for backpropagation
        self.inputs = inputs
        self.drive = np.sum(np.multiply(self.weights, inputs))
        self.output = sigmoid(self.drive+self.bias)
        return self.output
        

    def update(self, delta):
        # update all weights with the one delta multiplied by the matching input/activation multiplied with the learning rate (alpha)
        # notation on the homework sheet is not that clear but I think updates are like this:
        self.weights -= self.alpha * delta * self.inputs
        self.bias -= self.alpha * delta * 1.0