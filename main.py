"""Main file where we train the NN with the dataset"""

# Import the dataset to experiment with
from nnfs.datasets import spiral_data

# Import the NN utils
from layers.dense import DenseLayer

# Load data
X_train, y_train = spiral_data(samples=100, classes=3)

# Create a dense layer with 2 inputs and 3 neurons
dense = DenseLayer(2, 3)
dense.forward(X_train) # Perform a forward pass

# Print the output
print(dense.output[:5])
