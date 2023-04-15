"""Main file where we train the NN with the dataset"""

# Utils from the book
import nnfs
from nnfs.datasets import spiral_data

# Import the NN utils
from activation import ReLU, Softmax
from layers.dense import Dense

# Init the configuration of nnfs
nnfs.init()

# Load data
X_train, y_train = spiral_data(samples=100, classes=3)

# Create a dense layer with 2 inputs and 3 neurons
dense1 = Dense(2, 3)
dense1.forward(X_train) # Perform a forward pass

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
print(activation2.output[:5])
