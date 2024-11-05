import sys
import os
import torch

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

sys.path.insert(0,f'{DIR}/../')

import py487

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
np.random.seed(4)
X, t = datasets.make_blobs(n_samples=100, centers=3, n_features=4, center_box=(0, 10))

def test_1():
    torch_layer = torch.nn.Linear(4,10)
    W,b = list(torch_layer.parameters())
    W = W.detach().numpy().tolist()
    b = b.detach().numpy().tolist()
    solution = torch_layer(torch.Tensor(X[:4])).detach().numpy()
    
    layer = py487.nn.Linear(4,10)
    layer.W = W
    layer.b = b
    answer = np.array(layer.transform(X[:4]))

    assert np.all(np.abs(solution-answer) <= 0.0001)
    
def test_2():
    import torch.nn as nn
    # With square kernels and equal stride
    m = nn.Conv2d(16, 2, 3, stride=1)
    solution = np.array(m.weight.shape)
    
    layer = py487.nn.Conv2d(16, 2, 3)
    answer = np.array(np.array(layer.W).shape)
    
    assert np.all(np.abs(solution-answer) == 0)
    
def test_3():
    import torch.nn as nn
    input = torch.randn(20, 16, 32, 32)
    # With square kernels and equal stride
    m = nn.Conv2d(16, 2, 3, padding=1)
    solution = np.array(m(input).detach().numpy().shape)
    
    layer = py487.nn.Conv2d(16, 2, 3)
    answer = np.array(np.array(layer.transform_dryrun(input)).shape)
    
    assert np.all(np.abs(solution-answer) == 0)
    
def test_4():
    import torch.nn as nn
    input = torch.randn(20, 16, 32, 32)
    # With square kernels and equal stride
    m = nn.Conv2d(16, 2, 3, padding=1,bias=False)
    
    solution = np.array(m(input).detach().numpy())
    
    layer = py487.nn.Conv2d(16, 2, 3)
    layer.W = m.weight.detach().numpy().tolist()
    answer = np.array(layer.transform(input))
    
    assert np.all(np.abs(solution-answer) <= 1e-4)