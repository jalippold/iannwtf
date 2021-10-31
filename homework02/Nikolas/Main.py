import numpy as np
from MLP import MLP

# create a MLP with two inputs, one hidden layer with 4 perceptrons and one output neuron
mlp = MLP([2, 4, 1])

input = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
and_label = np.array([0., 0., 0., 1.])
or_label = np.array([0., 1., 1., 1.])
nand_label = np.array([1., 1., 1., 0.])
nor_label = np.array([1., 0., 0., 0.])
xor_label = np.array([0., 1., 1., 0.])
labels = np.array([
    and_label,
    or_label,
    nand_label,
    nor_label,
    xor_label
])

for i in range(1000):
    output = mlp.forward_step(input[0])
    mlp.backprop_step(labels[0:1,0])
    output = mlp.forward_step(input[1])
    mlp.backprop_step(labels[0:1,1])
    output = mlp.forward_step(input[2])
    mlp.backprop_step(labels[0:1,2])
    output = mlp.forward_step(input[3])
    mlp.backprop_step(labels[0:1,3])

output = mlp.forward_step(input[0])
print(f"Output for input {input[0]}: {output}")
output = mlp.forward_step(input[1])
print(f"Output for input {input[1]}: {output}")
output = mlp.forward_step(input[2])
print(f"Output for input {input[2]}: {output}")
output = mlp.forward_step(input[3])
print(f"Output for input {input[3]}: {output}")