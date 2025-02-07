import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights=np.random.randn(output_size,input_size)
        self.biases=np.random.randn(output_size,1)
        
    def forward(self, input):
        self.input=input
        self.output=np.dot(self.weights,input)+self.biases
        
        return self.output
    
    def backward(self, grad_output,learning_rate):
        grad_weights=np.dot(grad_output,self.input.T)
        grad_input=np.dot(self.weights.T,grad_output)
        grad_biases=grad_output
        
        self.weights-=learning_rate*grad_weights
        self.biases-=learning_rate*grad_biases
        
        return grad_input
    
    
        