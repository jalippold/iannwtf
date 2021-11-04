import numpy as np
from MathUtil import *

class Perceptron():
    def __init__(self, input_units):
        """
        Perceptron with bias, weights and ingoing activations.
        :param input_units is the number of inputs going into this perceptron
        """
        self.bias = np.random.randn()
        self.weights = np.array(np.random.randn(input_units))
        self.alpha = 1.0    # according to homework sheet this should be 1.0
        self.output = None
        self.inputs = None

    def forward_step(self, inputs):
        """
        Calculates the output of this perceptron.
        :param inputs are the activations going into this perceptron
        """
        if len(inputs) is not len(self.weights):
            raise IndexError(f"Length of input ({len(inputs)}) does not match length of weights ({len(self.weights)})!")
        # remember input/activations for backpropagation
        self.inputs = inputs
        self.output = sigmoid(np.sum(np.multiply(self.weights, inputs))+self.bias)
        return self.output
        

    def update(self, delta):
        """
        Update all weights and bias with the one delta multiplied by the matching input/activation multiplied with the learning rate (alpha).
        :param delta is the value of the delta-rule
        """
        # notation on the homework sheet is not that clear but I think updates are like this:
        self.weights -= self.alpha * delta * self.inputs
        self.bias -= self.alpha * delta * 1.0