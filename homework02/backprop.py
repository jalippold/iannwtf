import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidprime(x):
    sig = sigmoid(x)
    return sig*(1-sig)


class Perceptron():
    def __init__(self, input_units):
        """Perceptron with bias and i ingoing weighted activations"""
        self.bias = np.random.randn()
        self.weights = np.array([np.random.randn() for _ in range(input_units)])
        self.alpha = 1.
        self.output = None
        self.inputs = None
        self.drive = None

    def forward_step(self, inputs):
        if len(inputs) is not len(self.weights):
            raise IndexError("Length of input does not match length of weights!")
        # remember input/activations for backpropagation
        self.inputs = inputs
        #self.drive = np.sum(np.multiply(self.weights, inputs))
        #self.output = sigmoid(self.drive+self.bias)
        self.output = sigmoid(np.sum(np.multiply(self.weights, inputs))+self.bias)
        return self.output
        

    def update(self, delta):
        # update all weights with the one delta multiplied by the matching input/activation
        # notation on the homework sheet is not that clear but I think updates are like this:
        self.weights -= (self.alpha * delta * self.inputs)
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
        # output of input layer
        next_vals = np.array([p.forward_step(input) for p in self.input_layer])
        
        # loop through hidden layers
        for i in range(len(self.hidden_layers)):
            vals = np.copy(next_vals)
            next_vals = np.array([p.forward_step(vals) for p in self.hidden_layers[i]])
        
        # return output of output layer
        return np.array([p.forward_step(next_vals) for p in self.output_neurons])

    def backpropagate(self, output, expected_vals):
        # Use loss function and backpropagation to change Perceptrons weights
        # calculate delta for every perceptron and call p.update(delta)
        deltas = np.empty(len(self.output_neurons))
        
        # for output layer: delta = -(ti-yi)*sig'(diN)
        efunc = -1. * (expected_vals-output)
        for i in range(len(self.output_neurons)):
            deltas[i] = efunc[i]*sigmoidprime(self.output_neurons[i].output)
            self.output_neurons[i].update(deltas[i])

        # for other layers
        nextlayer = self.output_neurons
        nextsize = len(self.output_neurons)
        for l in range(len(self.hidden_layers)-1, -1, -1):
            nextdeltas = np.empty(len(self.hidden_layers[l]))
            
            for i in range(len(self.hidden_layers[l])):
                esum = 0
                for k in range(nextsize):
                    esum += deltas[k] * nextlayer[k].weights[i]

                nextdeltas[i] = esum * sigmoidprime(self.hidden_layers[l][i].output)
                self.hidden_layers[l][i].update(nextdeltas[i])

            nextsize = len(self.hidden_layers[l])
            nextlayer = self.hidden_layers[l]

            deltas = nextdeltas.copy()

        # do this again for the input layer!
        for i in range(len(self.input_layer)):
            esum = 0
            for k in range(len(self.hidden_layers[0])):
                esum += deltas[k] * self.hidden_layers[0][k].weights[i]
            
            delta = esum * sigmoidprime(self.input_layer[i].output)
            self.input_layer[i].update(delta)


    def __str__(self):
        ostr = ""
        for i in range(self.input_units):
            ostr += f"\tIn {i}"
        
        ostr += "\n\n"
        for i in range(len(self.input_layer)):
            ostr += f"P{i} {self.input_layer[i].output:.2f}/{self.input_layer[i].bias:.2f}\t"


        ostr += "\n\n"
        for i in range(len(self.hidden_layers)):
            for j in range(len(self.hidden_layers[i])):
                ostr += f"P{j} {self.hidden_layers[i][j].output:.2f}/{self.hidden_layers[i][j].bias:.2f}\t"


            ostr += "\n\n"

        for i in range(len(self.output_neurons)):
            ostr += f"\tOut {i}"
        
        return ostr



# MLP with 2 inputs, an input layer with 4 perceptrons, one hidden layer with 4 perceptrons and an output layer with a single perceptron
mlp = MLP(3, [10, 10], 5)

inpt = np.array([[0,0, 0*0],[0,1, 0*1],[1,0, 1*0],[1,1, 1*1]])
expected_output = np.array([[0, 0, 1, 1, 0],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 1, 0, 0, 0]])

accuracy = [ [ [] for _ in range(5) ] for _ in range(4)]
loss = [ [ [] for _ in range(5) ] for _ in range(4)]

for i in range(1000):
    for i in range(4):
        output = mlp.calculate(inpt[i])
        for j in range(len(output)):
            loss[i][j].append((expected_output[i][j]-output[j])**2)
            if expected_output[i][j] == int(round(output[j], 0)):
                accuracy[i][j].append(1)
            else:
                accuracy[i][j].append(0)

        mlp.backpropagate(output, expected_output[i])
    

np.set_printoptions(precision=3, suppress=True)
print(f"Loss (Input/Logic function):\n {np.mean(loss, axis=2)}\n")
print(f"Accuracy(Input/Logic function):\n {np.mean(accuracy, axis=2)}")