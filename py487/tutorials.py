from torch import nn

# We define our neural network by subclassing `nn.Module`, and initialize
# the neural network layers in `__init__`. Every `nn.Module` subclass
# implements the operations on input data in the `forward` method.

# You can get the answers and what code to insert here: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Your solution here

    def forward(self, x):
        # Your solution here
        return logits