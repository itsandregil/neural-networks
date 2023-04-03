"""
Implementation of a layer in pure Python,
this is how neurons and layers work.
"""

# Create a layer with 3 neurons
inputs = [1, 2, 3, 4]
weights = [
	[0.2, 0.8, -0.5],
	[0.4, 0.3, 0.6],
	[-0.2, -0.7, 0.9]
]
bias = [2, 3, 0.5]
layer_outputs = [] # Output of the current layer

# Calculate the output for each neuron
for neuron_weights, neuron_bias in zip(weights, bias):
	neuron_output = 0
	for n_input, weight in zip(inputs, neuron_weights):
		# Multiply every input with the neuron's weight
		neuron_output += n_input * weight
	neuron_output += bias
	layer_outputs.append(neuron_output)
