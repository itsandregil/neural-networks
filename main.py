"""Main file where we train the NN with the dataset"""

# Utils from the book
import nnfs
from nnfs.datasets import spiral_data

# Import the NN utils
from activation.relu import ReLU
from layers.dense import Dense

# Init the configuration of nnfs
nnfs.init()

# Load data
X_train, y_train = spiral_data(samples=100, classes=3)

# Create a dense layer with 2 inputs and 3 neurons
dense = Dense(2, 3)
dense.forward(X_train) # Perform a forward pass

# Create an activation function
activation = ReLU()
activation.forward(dense.output) 

# Print the output
print(activation.output[:5])
