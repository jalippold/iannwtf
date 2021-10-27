import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidprime(x):
    sig = sigmoid(x)
    return sig*(1-sig)


# We will train the network on logical gates, means there are two (sometimes only one) inputs and an output for each functionality
# and, or, not and, not or, xor => 5 outputs
# as training data we will need the four possible input combinations and the corresponding five output values

training_data = {
    # input: [and, or, nand, nor, xor]
    (0, 0): [0, 0, 1, 1, 0],
    (0, 1): [0, 1, 1, 0, 1],
    (1, 0): [0, 1, 1, 0, 1],
    (1, 1): [1, 1, 0, 0, 0]
}

class Perceptron():
    def __init__(self, input_units):
        """Perceptron with bias and i ingoing weighted activations"""
        self.bias = np.random.randn()
        self.weights = [np.random.randn() for _ in range(input_units)]
        self.alpha = 1

    def forward_step(self, inputs):
        pass

    def update(self, delta):
        pass


class MLP():
    def __init__(self, input_units, layers, output_units):
        """layers is an array, containing the input and hidden layers
            It's length-1 is the number of hidden layers, the elements are the number of neurons on every layer"""
        self.input_units = input_units
        self.input_layer = []
        self.hidden_layers = []
        self.output_neurons = []

        # initialize input layer
        for _ in range(layers[0]):
            self.input_layer.append(Perceptron(input_units))

        # initialize hidden layers
        for i in range(1, len(layers)):
            self.hidden_layers.append([])
            for j in range(layers[i]):
                self.hidden_layers[i-1].append(Perceptron(layers[i-1]))

        # initialize output layer
        for _ in range(output_units):
            self.output_neurons.append(Perceptron(layers[len(layers)-1]))


    def calculate(self, input):
        if len(input) is not self.input_units:
            raise ValueError("Len of input does not match networks structure!")

        vals = []
        next_vals = []

        for i in len(self.input_layer):
            next_vals.append(self.input_layer[i].forward_step(input))
        

        for i in range(len(self.hidden_layers)):
            vals = next_vals.copy()

            if len(vals) is not len(self.hidden_layers[i]):
                raise IndexError("Somethings wrong in the internal structure!")

            next_vals.clear()
            for j in range(self.hidden_layers[i]):
                next_vals.append(self.hidden_layers[i][j].forward_step(vals))

        if len(next_vals) is not len(self.output_neurons):
            raise IndexError("Somethings wrong in the internal structure!")

        outputs = []
        for i in range(len(self.output_neurons)):
            outputs.append(self.output_neurons[i].forward_step(next_vals))

        return outputs

    def backpropagate(self, output, expected_vals):
        # Use loss function and backpropagation to change Perceptrons weights
        pass

# MLP with 2 inputs, an input layer with 4 perceptrons, one hidden layer with 4 perceptrons and an output layer with a single perceptron
mlp = MLP(2, [4, 4], 1)