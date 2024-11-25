import random
import numpy as np

import math

import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # You must extract and use the weights that are randomly generated in the following data structures
        torch.manual_seed(0)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def convert_output(self,out,list_output):
        if isinstance(out, torch.Tensor) and list_output:
            out = out.tolist()
        elif isinstance(out, np.ndarray) and list_output:
            out = out.tolist()
        return out
        
    def scaled_dot_product_attention(self, Q, K, V, list_output=True):
        # Your solution here. 
        out = []
        return self.convert_output(out,list_output)
        
    # I implemented helper functions such as matmul and tranpose. I suggest you do the same to make incremental progress
    
    def split_heads(self, x, list_output=True):
        output = []
        # Your solution here.
        return self.convert_output(output,list_output)
        
    def combine_heads(self, x, list_output=True):
        output = None
        # Your solution here. 
        return self.convert_output(output,list_output)
    
    def forward(self, Q, K, V, list_output=True):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V)
        output = self.W_o(torch.from_numpy(np.array(self.combine_heads(attn_output))).float())
        return self.convert_output(output,list_output)
