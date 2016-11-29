# Based off of bpnn by Neil Schemenauer <nas@arctrix.com>
# Modifications and second layer added by Charles Timmerman <cttimm4427@ung.edu>
# Written in Python. See http://www.python.org
# Placed in the public domain.

import math
import random
import string
import json

random.seed(0)

def rand(min, max):
    return (max-min)*random.random() + min

# Creates a matrix with append
def fill_matrix(I, J, fill=0.0):
    matrix = []
    for i in range(I):
        matrix.append([0.0]*J)
    return matrix

#Sigmoid function for activation (tanh)
def activationFunction(x):
    return math.tanh(x)

#Dx of Sigmoid activation function used for calculating error
def dxactivationFunction(y):
    return 1.0 - y**2

class NN:
    def __init__(self, n_input = 8, n_layer1 = 7,  n_layer2 = 7, n_output = 1):
        self.n_input = n_input + 1
        self.n_layer1 = n_layer1
        self.n_layer2 = n_layer2
        self.n_output = n_output
        # Activations for nodes
        self.a_input = [1.0] * self.n_input
        self.a_layer1 = [1.0] * self.n_layer1
        self.a_layer2 = [1.0] * self.n_layer2
        self.a_output = [1.0] * self.n_output
        # Weights
        self.input_weights = fill_matrix(self.n_input, self.n_layer1)
        self.hidden_weights = fill_matrix(self.n_layer1, self.n_layer2)
        self.output_weights = fill_matrix(self.n_layer2, self.n_output)
        # Set the weight matrices to random values
        for i in range(self.n_input):
            for j in range(self.n_layer1):
                self.input_weights[i][j] = rand(-(1/n_input),(1/n_input))
        for j in range(self.n_layer1):
            for k in range(self.n_layer2):
                self.hidden_weights[j][k] = rand(-n_input,n_input)/math.sqrt(n_input)
        for j in range(self.n_layer2):
            for k in range(self.n_output):
                self.output_weights[j][k] = rand(-n_layer1, n_layer1)/math.sqrt(n_layer1)  
        # Momentum; change in weights
        self.c_input = fill_matrix(self.n_input, self.n_layer1)
        self.c_output = fill_matrix(self.n_layer2, self.n_output)
        self.c_hidden = fill_matrix(self.n_layer1, self.n_layer2)
        
    # Activations for input, hidden, output
    def update(self, inputs):
        if len(inputs) != self.n_input - 1:
            raise ValueError("Wrong number of inputs")
        
        # Input activations
        for i in range(self.n_input-1):
            self.a_input[i] = inputs[i]
            # self.a_input[i] = activationFunction(inputs[i])

        # Hidden activations
        for j in range(self.n_layer1):
            sum = 0.0
            for i in range(self.n_input):
                sum = sum + self.a_input[i] * self.input_weights[i][j]
            self.a_layer1[j] = activationFunction(sum)
                
        for j in range(self.n_layer2):
            sum = 0.0
            for i in range(self.n_layer1):
                sum = sum + self.a_layer1[i] * self.hidden_weights[i][j]
            self.a_layer2[j] = activationFunction(sum)

         # Output activations
        for k in range(self.n_output):
            sum = 0.0
            for j in range(self.n_layer2):
                sum = sum + self.a_layer2[j] * self.output_weights[j][k]
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

        # Calculate Second Layer Error
        hidden2_deltas = [0.0] * self.n_layer2
        for j in range(self.n_layer2):
            error = 0.0
            for k in range(self.n_output):
                error = error + output_deltas[k] * self.output_weights[j][k]
            hidden2_deltas[j] = dxactivationFunction(self.a_layer2[j]) * error

        # Calculate First Layer Error
        hidden_deltas = [0.0] * self.n_layer1
        for j in range(self.n_layer1):
            error = 0.0
            for k in range(self.n_layer2):
                error = error + hidden2_deltas[k] * self.hidden_weights[j][k]
            hidden_deltas[j] = dxactivationFunction(self.a_layer1[j]) * error

        # Update Output Weights
        for j in range(self.n_layer2):
            for k in range(self.n_output):
                delta = output_deltas[k] * self.a_layer2[j]
                self.output_weights[j][k] = self.output_weights[j][k] + (N * delta) + (M * self.c_output[j][k])
                self.c_output[j][k] = delta
        
        # Update Hidden Weights
        for j in range(self.n_layer1):
            for k in range(self.n_layer2):
                delta = hidden2_deltas[k] * self.a_layer1[j]
                self.hidden_weights[j][k] = self.hidden_weights[j][k] + (N * delta) + (M * self.c_hidden[j][k])
                self.c_hidden[j][k] = delta

        # Update Input Weights
        for i in range(self.n_input):
            for j in range(self.n_layer1):
                delta = hidden_deltas[j] * self.a_input[i]
                self.input_weights[i][j] = self.input_weights[i][j] + (N * delta) + (M * self.c_input[i][j])
                self.c_input[i][j] = delta

        # Calculate Error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k] - self.a_output[k])**2
        return error
    
    # Test the class with a pattern
    def test(self, patterns):
        for p in patterns:
            return self.update(p[0])[0]

    # Print the weights
    def weights(self):
        return [self.input_weights, self.hidden_weights, self.output_weights]

    # Train the NN, N = learning rate, M = momentum factor
    def train(self, patterns, iterations, N, M):
        print("Iterations: %d, N: %f, M: %f" % (iterations, N, M))
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backProp(targets, N, M)
            if i % 200 == 0:
                print("Error %-.5f" % error)
    