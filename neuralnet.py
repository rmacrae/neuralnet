# Ryan MacRae
# 3/24/2017
# Neural Network for XOR


import numpy as np

# sigmoid activation function
def nonlin(x, deriv):
    if (deriv == True):
        return x * (1 - x)
    else: return 1 / (1 + np.exp(-x))

# number of epochs.
epochs = 1000000

print ("Neural Network for XOR\n")

# input values
X = np.array([[0, 0], [0, 1], [1, 0],[1, 1]])

# output values
y = np.array([ [0], [1], [1], [0] ])

# initialize weights randomly with mean 0
synapse0 = np.random.random(size=(2, 3)) - 1    # Weights on hidden layer inputs
synapse1 = np.random.random(size=(3, 1)) - 1    # Weights on output layer inputs

for iter in range(epochs):
    # forward propagation
    layer_0 = X
    layer_1 = nonlin(np.dot(layer_0, synapse0), False)
    layer_2 = nonlin(np.dot(layer_1, synapse1), False)

    # Error Calculation
    l2_error = y - layer_2

    # backward propagation
    l2_delta = l2_error * nonlin(layer_2, True)
    l1_delta = np.dot(l2_delta, synapse1.T) * nonlin(layer_1, True)

    # update weights
    synapse1 += np.dot(layer_1.T, l2_delta)
    synapse0 += np.dot(layer_0.T, l1_delta)

# print results
print ("Output After Training (%d epochs):" % epochs)
print (layer_2)
print ("\n")
print ("Expected Values: ")
print (y)

