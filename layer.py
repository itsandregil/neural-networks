"""Implementation of a layer with numpy"""

import numpy as np

# Create a layer with 3 neurons
inputs = [1, 2, 3]
weights = [
	[0.2, 0.8, -0.5],
	[0.4, 0.3, 0.6],
	[-0.2, -0.7, 0.9]
]
bias = [2, 3, 0.5]

# Remember the dot product rule
# Where the neighboring dimension must be equal.
layer_outputs = np.dot(weights, inputs) + bias # Output of the current layer

print(layer_outputs)
