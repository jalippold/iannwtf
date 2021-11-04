import numpy as np
from MathUtil import *
from Perceptron import Perceptron

class MLP():
    def __init__(self, layers):
        """
        layers is an array, containing input, hidden and output layers.
        It's length-2 is the number of hidden layers.
        The elements are the number of neurons on every layer
        """
        # input layer is not really made up of perceptrons but instead only describes the input vector
        # still remember how many inputs there are
        self.input_number = layers[0]

        # initialize hidden layers
        hlayers = []
        for i in range(1, len(layers)-1):
            hlayers.append([])
            for j in range(layers[i]):
                hlayers[-1].append(Perceptron(layers[i-1]))
        self.hidden_layers = np.array(hlayers)

        # initialize output layer
        self.output_neurons = np.array([Perceptron(layers[-2]) for _ in range(layers[-1])])

        self.output = np.empty(layers[-1])


    def forward_step(self, input):
        next_vals = input
        # loop through hidden layers
        for hl in self.hidden_layers:
            vals = np.copy(next_vals)
            next_vals = np.array([p.forward_step(vals) for p in hl])
        
        # return output of output layer
        self.output = np.array([p.forward_step(next_vals) for p in self.output_neurons])
        return self.output

    def backprop_step(self, expected_vals):
        # Use loss function and backpropagation to change Perceptrons weights
        # calculate delta for every layer and call p.update(delta)
        deltas = np.empty(len(self.output_neurons))
        
        # for output layer: delta = -(ti-yi)*sig'(diN)
        efunc = -1.0 * (expected_vals-self.output)
        for i, n_out in enumerate(self.output_neurons):
            deltas[i] = efunc[i] * sigmoidprime(n_out.output)
            n_out.update(deltas[i])

        # for other layers
        nextlayer = self.output_neurons
        nextsize = len(self.output_neurons)
        for l in range(len(self.hidden_layers)-1, -1, -1):
            nextdeltas = np.empty(len(self.hidden_layers[l]))
            # for each perceptron in this hidden layer
            for i in range(len(self.hidden_layers[l])):
                esum = 0
                for k in range(nextsize):
                    esum += deltas[k] * nextlayer[k].weights[i]

                nextdeltas[i] = esum * sigmoidprime(self.hidden_layers[l][i].output)
                self.hidden_layers[l][i].update(nextdeltas[i])

            nextsize = len(self.hidden_layers[l])
            nextlayer = self.hidden_layers[l]

            deltas = nextdeltas.copy()


    def __str__(self):
        ostr = ""
        for i in range(self.input_number):
            ostr += f"\tIn {i}"


        ostr += "\n\n"
        for i in range(len(self.hidden_layers)):
            for j in range(len(self.hidden_layers[i])):
                ostr += f"P{j} {self.hidden_layers[i][j].output}/{self.hidden_layers[i][j].bias}\t"


            ostr += "\n\n"

        for i in range(len(self.output_neurons)):
            ostr += f"\tOut {i}"
        
        return ostr