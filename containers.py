"""Types of data container that are often used
when working with NNs"""

import numpy as np

# List
data = [1, 3, 4, 5]

# Vector
vec = np.array(data)

# Row vector
rvec = np.array([data])

# Expand a dimension of the vector data
# to create a row vector
rv = np.expand_dims(np.array(data), axis=0)
print(rv)