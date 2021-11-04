"""
This is the main script for executing the self-implemented MLP on learning logical operations.
"""
import numpy as np
from MLP import MLP
from MathUtil import calc_mean_accuracy, calc_mean_loss
import matplotlib.pyplot as plt

# create a MLP with two inputs, one hidden layer with 4 perceptrons and one output neuron
mlp = MLP([2, 4, 1])

analysis = []

input = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
and_label = np.array([0., 0., 0., 1.])
or_label = np.array([0., 1., 1., 1.])
nand_label = np.array([1., 1., 1., 0.])
nor_label = np.array([1., 0., 0., 0.])
xor_label = np.array([0., 1., 1., 0.])

target = xor_label

for i in range(1000):
    outputs = []
    for j in range(4):
        outputs.append(mlp.forward_step(input[j]))
        mlp.backprop_step(target[j])
    
    # keep track of accuracy and loss for later visualization
    mean_accuracy = calc_mean_accuracy(outputs, target)
    mean_loss = calc_mean_loss(outputs, target)
    analysis.append([i, mean_accuracy, mean_loss])

analysis = np.array(analysis)
fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(analysis[:,0], analysis[:,1])
ax1.set_ylabel("Average Accuracy")

ax2.plot(analysis[:,0], analysis[:,2])
ax2.set_ylabel("Average Loss")
ax2.set_xlabel("Epoch")

plt.show()

print("Doing one test:")
output = mlp.forward_step(input[0])
print(f"Output for input {input[0]}: {output}")
output = mlp.forward_step(input[1])
print(f"Output for input {input[1]}: {output}")
output = mlp.forward_step(input[2])
print(f"Output for input {input[2]}: {output}")
output = mlp.forward_step(input[3])
print(f"Output for input {input[3]}: {output}")