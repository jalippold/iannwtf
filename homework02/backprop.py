import numpy as np

def sigmoid(x):
    pass

def sigmoidprime(x):
    pass


# We will train the network on logical gates, means there are two (sometimes only one) inputs and an output for each functionality
# and, or, not and, not or, xor => 5 outputs
# as training data we will need the four possible input combinations and the corresponding five output values

training_data = {
    # input: [and, or, nand, nor, xor]
    [0, 0]: [0, 0, 1, 1, 0],
    [0, 1]: [0, 1, 1, 0, 1],
    [1, 0]: [0, 1, 1, 0, 1],
    [1, 1]: [1, 1, 0, 0, 0]
}