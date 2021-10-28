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
        self.weights = np.array([np.random.randn() for _ in range(input_units)])
        self.alpha = 1
        self.output = None
        self.inputs = None

    def forward_step(self, inputs):
        if len(inputs) is not len(self.weights):
            raise IndexError("Length of input does not match length of weights!")
        # remember input/activations for backpropagation
        self.inputs = inputs
            
        self.output = sigmoid(np.sum(np.multiply(self.weights, inputs))+self.bias)
        return self.output
        

    def update(self, delta):
        # update all weights with the one delta multiplied by the matching input/activation
        # notation on the homework sheet is not that clear but I think updates are like this:
        self.weights += (delta * self.inputs)
        self.bias -= (self.alpha * delta)


class MLP():
    def __init__(self, input_units, layers, output_units):
        """layers is an array, containing the input and hidden layers
            It's length-1 is the number of hidden layers, the elements are the number of neurons on every layer"""
        self.input_units = input_units

        # layers[0] represents size of input layer
        self.input_layer = np.array([Perceptron(input_units) for _ in range(layers[0])])

        # initialize hidden layers
        hlayers = []
        for i in range(1, len(layers)):
            hlayers.append([])
            for j in range(layers[i]):
                hlayers[i-1].append(Perceptron(layers[i-1]))
        self.hidden_layers = np.array(hlayers)

        # initialize output layer
        self.output_neurons = np.array([Perceptron(layers[-1]) for _ in range(output_units)])


    def calculate(self, input):
        next_vals = np.array([p.forward_step(input) for p in self.input_layer])
        

        for i in range(len(self.hidden_layers)):
            vals = np.copy(next_vals)

            next_vals = np.array([p.forward_step(vals) for p in self.hidden_layers[i]])

        return np.array([p.forward_step(next_vals) for p in self.output_neurons])

    def backpropagate(self, output, expected_vals):
        # Use loss function and backpropagation to change Perceptrons weights
        # calculate delta for every perceptron and call p.update(delta)

        pass

    def __str__(self):
        ostr = ""
        for i in range(self.input_units):
            ostr += f"\tIn {i}"
        
        ostr += "\n\n"
        for i in range(len(self.input_layer)):
            ostr += f"Val {self.input_layer[i].output:.2f}\t"


        ostr += "\n\n"
        for i in range(len(self.hidden_layers)):
            for j in range(len(self.hidden_layers[i])):
                ostr += f"Val {self.hidden_layers[i][j].output:.2f}\t"


            ostr += "\n\n"

        for i in range(len(self.output_neurons)):
            ostr += f"\tOut {i}"
        
        return ostr



# MLP with 2 inputs, an input layer with 4 perceptrons, one hidden layer with 4 perceptrons and an output layer with a single perceptron
mlp = MLP(2, [4, 4], 1)

output = mlp.calculate(np.array([0,0]))
print(output)
output = mlp.calculate(np.array([0,1]))
print(output)
output = mlp.calculate(np.array([1,0]))
print(output)
output = mlp.calculate(np.array([1,1]))
print(output)
print()

print(mlp)