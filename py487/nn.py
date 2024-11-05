import random
import numpy as np

import math

import torch.nn as nn
import torch

def get_new_random_weight():
    return random.random()

class Linear:
    def __init__(self,in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        # initialize weights and biases randomly
        self.W = []
        self.b = []
        random.seed(42)
        self.init_parameters()
        
    def init_parameters(self):
        # here you should initialize the parameters by calling get_new_random_weight(). Reminder to do the biases at the end.
        # weights of shape (out, in)
        self.W = None
        self.b = None
        #print(self.W, self.b)

    def transform(self,X):
        out = None
        return out
    
class Conv2d:
    def __init__(self,in_channels, out_channels, kernel_size=3):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.W = []
        random.seed(42)
        # Your solution here
    
    def transform_dryrun(self,X):
        out = []
        # Your solution here
        return out
    
    def transform(self,X):
        out = self.transform_dryrun(X) 
        # Your solution here
        return out
    