"""Main file where we train the NN with the dataset"""

# Utils from the book
import nnfs
import numpy as np
from nnfs.datasets import spiral_data

# Import the NN utils
from activation import ReLU, Softmax
from layers.dense import Dense
from loss import CategoricalCrossEntropy

# Init the configuration of nnfs
nnfs.init()

# Load data
X_train, y_train = spiral_data(samples=100, classes=3)

# Create a dense layer with 2 inputs and 3 neurons
dense1 = Dense(2, 3)
dense1.forward(X_train)  # Perform a forward pass

# Create an activation function
activation1 = ReLU()
activation1.forward(dense1.output)

# Add a new dense layer with 3 inputs and 3 neurons
dense2 = Dense(3, 3)
dense2.forward(activation1.output)

# Add an softmat activation function since we want to
# solve a multiple classificacion problem
activation2 = Softmax()
activation2.forward(dense2.output)
print(activation2.output[:10])

# Create a loss function
loss_function = CategoricalCrossEntropy()
# Calculate the loss using the softmax activation function layer
loss = loss_function.calculate(activation2.output, y_train)
print(f"loss: {loss}")

# Calculate the Accurary of the Model
# Accuracy: Mean value of how many times the prediction was correct
predictions = np.argmax(activation2.output, axis=1)

if len(y_train.shape) == 2:
    y_train = np.argmax(y_train, axis=1)

accuracy = np.mean(predictions == y_train)
print(f"Acc: {accuracy}")