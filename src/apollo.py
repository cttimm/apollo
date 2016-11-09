# Based off of bpnn by Neil Schemenauer <nas@arctrix.com>
# Written in Python. See http://www.python.org
# Placed in the public domain.
# 

import math
import random
import string

random.seed(0)

def rand(min, max):
    return (max-min)*random.random() + min

def makeMatrix(I, J, fill=0.0):
    pass

def sigmoid(x):
    pass

def dsigmoid(y):
    pass

class NN:
    def __init__(self, n_input, n_hidden, n_output):
        self.n_input = n_input + 1
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Activations for nodes
        self.a_input = [1.0]*self.n_input
        self.a_hidden = [1.0]*self.n_hidden
        self.a_output = [1.0]*self.n_output

        # Weights
        self.w_input = makeMatrix(self.n_input, self.n_hidden)
        self.w_output = makeMatrix(self.n_hidden, self.n_output)

        # Set the weight matrices to random values
        for i in range(self.n_input):
            for j in range(self.n_hidden):
                self.w_input[i][j] = rand(-0.2, 0.2)
        for j in range(self.n_hidden):
            for k in range(self.n_output):
                self.w_output[j][k] = rand(-2.0, 2.0)

        # Momentum; change in weights
        self.c_input = makeMatrix(self.n_input, self.n_hidden)
        self.c_output = makeMatrix(self.n_hidden, self.n_output)
    
    # Activations for input, hidden, output
    def update(self, inputs):
        pass

    # Calculate error (output, hidden), update input and output weights
    def backProp(self, targets, N, M):
        pass
    
    # Test the class with a pattern
    def test(self, patterns):
        pass

    # Print the weights
    def weights(self):
        pass

    # Train the NN, N = learning rate, M = momentum factor
    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        pass
    