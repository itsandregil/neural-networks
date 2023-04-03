"""Implement a layer that takes inputs in a batch"""

import numpy as np

# Input and Weights
# Create a random matrix of inputs
inputs = [
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]
]
weigths = [
	[0.2, 0.8, -0.5, 1],
	[0.5, -0.91, 0.26, -0.5],
	[-0.26, -0.27, 0.17, 0.87]
]
# Biases
biases = [2, 3, 0.5]

outputs = np.dot(inputs, np.array(weigths).T) + biases
print(outputs)