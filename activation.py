import numpy as np
from layer import Layer

class Activation(Layer):
    def __init__(self, activation_function,activation_derivative):
        self.activation_derivative=activation_derivative
        self.activation_function = activation_function
        
    def forward(self, input):
        self.input = input
        self.output = self.activation_function(input)
        return self.output
    
    def backward(self, grad_output,learning_rate):
        return np.multiply(grad_output,self.activation_derivative(self.input))