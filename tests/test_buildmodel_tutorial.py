import sys
import os
sys.path.append("..")

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import joblib 
answers = joblib.load(str(DIR)+"/answers_buildmodel_tutorial.joblib")

# Import the student solutions
import py487

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
np.random.seed(1)
c = np.random.rand(6,4)
torch.manual_seed(1)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def test_exercise_1():
    model = py487.tutorials.NeuralNetwork().to(device)
    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    your_answer = pred_probab.tolist()
    assert np.all(answers['exercise_1'] == your_answer)