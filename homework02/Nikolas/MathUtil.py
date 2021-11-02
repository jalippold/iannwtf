import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoidprime(x):
    sig = sigmoid(x)
    return sig*(1.0-sig)

def calc_mean_accuracy(outputs, targets):
    if len(outputs) != len(targets):
        raise IndexError("Length of outputs does not match target length!")
    correct = 0
    for i, output in enumerate(outputs):
        if abs(output-targets[i]) < 0.5:
            correct += 1
    return float(correct)/float(len(outputs))

def calc_mean_loss(outputs, targets):
    if len(outputs) != len(targets):
        raise IndexError("Length of outputs does not match target length!")
    loss_sum = 0.0
    for i, output in enumerate(outputs):
        loss_sum += abs(output-targets[i])
    return float(loss_sum)/float(len(outputs))