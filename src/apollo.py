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

# Creates a matrix with append
def initMatrix(I, J, fill=0.0):
    matrix = []
    for i in range(I):
        matrix.append([0.0]*J)
    return matrix

#Sigmoid function for activation (tanh)
def activationFunction(x):
    return math.tanh(x)

#Dx of Sigmoid activation function
def dxactivationFunction(y):
    return 1.0 - y**2

class NN:
    def __init__(self, n_input, n_hidden, n_output):
        self.n_input = n_input + 1
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Activations for nodes
        self.a_input = [1.0] * self.n_input
        self.a_hidden = [1.0] * self.n_hidden
        self.a_output = [1.0] * self.n_output

        # Weights
        self.w_input = initMatrix(self.n_input, self.n_hidden)
        self.w_output = initMatrix(self.n_hidden, self.n_output)

        # Set the weight matrices to random values
        for i in range(self.n_input):
            for j in range(self.n_hidden):
                self.w_input[i][j] = rand(-0.2, 0.2)
        for j in range(self.n_hidden):
            for k in range(self.n_output):
                self.w_output[j][k] = rand(-2.0, 2.0)

        # Momentum; change in weights
        self.c_input = initMatrix(self.n_input, self.n_hidden)
        self.c_output = initMatrix(self.n_hidden, self.n_output)
    
    # Activations for input, hidden, output
    def update(self, inputs):
        if len(inputs) != self.n_input - 1:
            raise ValueError("Wrong number of inputs")
        
        # Input activations
        for i in range(self.n_input-1):
            self.a_input[i] = inputs[i]

        # Hidden activations
        for j in range(self.n_hidden):
            sum = 0.0
            for i in range(self.n_input):
                sum = sum + self.a_input[i] * self.w_input[i][j]
            self.a_hidden[j] = activationFunction(sum)
                
         # Output activations
        for k in range(self.n_output):
            sum = 0.0
            for j in range(self.n_hidden):
                sum = sum + self.a_hidden[j] * self.w_output[j][k]
            self.a_output[k] = activationFunction(sum) 
        return self.a_output[:]

    # Calculate error (output, hidden), update input and output weights
    # N = learning factor, M = momentum factor
    def backProp(self, targets, N, M):
        
        # Calculate Output Error
        output_deltas = [0.0] * self.n_output
        for k in range(self.n_output):
            error = targets[k] - self.a_output[k]
            output_deltas[k] = dxactivationFunction(self.a_output[k]) * error

        # Calculate Hidden Error
        hidden_deltas = [0.0] * self.n_hidden
        for j in range(self.n_hidden):
            error = 0.0
            for k in range(self.n_output):
                error = error + output_deltas[k] * self.w_output[j][k]
            hidden_deltas[j] = dxactivationFunction(self.a_hidden[j]) * error

        # Update Output Weights
        for j in range(self.n_hidden):
            for k in range(self.n_output):
                delta = output_deltas[k] * self.a_hidden[j]
                self.w_output[j][k] = self.w_output[j][k] + (N * delta) + (M * self.c_output[j][k])
                self.c_input[j][k] = delta

        # Update Input Weights
        for i in range(self.n_input):
            for j in range(self.n_hidden):
                delta = hidden_deltas[j] * self.a_input[i]
                self.w_input[i][j] = self.w_input[i][j] + (N * delta) + (M * self.c_input[i][j])
                self.c_input[i][j] = delta

        # Calculate Error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k] - self.a_output[k])**2
        return error
    
    # Test the class with a pattern
    def test(self, patterns):
        for p in patterns:
            print(p[0], ' --> ', self.update(p[0]))

    # Print the weights
    def weights(self):
        pass

    # Train the NN, N = learning rate, M = momentum factor
    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backProp(targets, N, M)
            if i % 100 == 0:
                print("Error %-.5f" % error)
    