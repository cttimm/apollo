# Based off of bpnn by Neil Schemenauer <nas@arctrix.com>
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
    def __init__(self, n_input = 8, n_hidden = 4,  n_hidden2 = 4, n_output = 1):
        self.n_input = n_input + 1
        self.n_hidden = n_hidden
        self.n_hidden2 = n_hidden2
        self.n_output = n_output

        # Activations for nodes
        self.a_input = [1.0] * self.n_input
        self.a_hidden = [1.0] * self.n_hidden
        self.a_hidden2 = [1.0] * self.n_hidden2
        self.a_output = [1.0] * self.n_output

        # Weights
        self.input_weights = initMatrix(self.n_input, self.n_hidden)
        self.hidden_weights = initMatrix(self.n_hidden, self.n_hidden2)
        self.output_weights = initMatrix(self.n_hidden2, self.n_output)

        # Set the weight matrices to random values
        for i in range(self.n_input):
            for j in range(self.n_hidden):
                self.input_weights[i][j] = rand(-0.2, 0.2)
        for j in range(self.n_hidden):
            for k in range(self.n_hidden2):
                self.hidden_weights[j][k] = rand(-2.0, 2.0)
        for j in range(self.n_hidden):
            for k in range(self.n_output):
                self.output_weights[j][k] = rand(-0.75, .75)

        # Momentum; change in weights
        self.c_input = initMatrix(self.n_input, self.n_hidden)
        self.c_output = initMatrix(self.n_hidden2, self.n_output)
        self.c_hidden = initMatrix(self.n_hidden, self.n_hidden2)
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
                sum = sum + self.a_input[i] * self.input_weights[i][j]
            self.a_hidden[j] = activationFunction(sum)
                
        for j in range(self.n_hidden2):
            sum = 0.0
            for i in range(self.n_hidden):
                sum = sum + self.a_hidden[i] * self.hidden_weights[i][j]
            self.a_hidden2[j] = activationFunction(sum)

         # Output activations
        for k in range(self.n_output):
            sum = 0.0
            for j in range(self.n_hidden2):
                sum = sum + self.a_hidden2[j] * self.output_weights[j][k]
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
        hidden2_deltas = [0.0] * self.n_hidden2
        for j in range(self.n_hidden2):
            error = 0.0
            for k in range(self.n_output):
                error = error + output_deltas[k] * self.output_weights[j][k]
            hidden2_deltas[j] = dxactivationFunction(self.a_hidden2[j]) * error

        # Calculate First Layer Error
        hidden_deltas = [0.0] * self.n_hidden
        for j in range(self.n_hidden):
            error = 0.0
            for k in range(self.n_hidden2):
                error = error + hidden2_deltas[k] * self.hidden_weights[j][k]
            hidden_deltas[j] = dxactivationFunction(self.a_hidden[j]) * error

        # Update Output Weights
        for j in range(self.n_hidden2):
            for k in range(self.n_output):
                delta = output_deltas[k] * self.a_hidden2[j]
                self.output_weights[j][k] = self.output_weights[j][k] + (N * delta) + (M * self.c_output[j][k])
                self.c_output[j][k] = delta
        
        # Update Hidden Weights
        for j in range(self.n_hidden):
            for k in range(self.n_hidden2):
                delta = hidden2_deltas[k] * self.a_hidden[j]
                self.hidden_weights[j][k] = self.hidden_weights[j][k] + (N * delta) + (M * self.c_hidden[j][k])
                self.c_hidden[j][k] = delta

        # Update Input Weights
        for i in range(self.n_input):
            for j in range(self.n_hidden):
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
            print(self.update(p[0])[0])

    # Print the weights
    def weights(self):
        return [self.input_weights, self.hidden_weights, self.output_weights]

    # Train the NN, N = learning rate, M = momentum factor
    def train(self, patterns, iterations=1000, N=0.00025, M=0.00025):
        print("Training neural network with samples")
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backProp(targets, N, M)
            if i % 200 == 0:
                print("Error %-.5f" % error)
    