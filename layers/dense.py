"""Implementation of a Dense Layer class"""

import numpy as np


class DenseLayer:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        # Initialize weigths and biases
        # We set the shape of weights as (inputs, neurons)
        # so we don't have to transpose everytime
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: int):
        """Calculate output values from inputs, weights and biases"""
        self.output = np.dot(inputs, self.weights) + self.biases