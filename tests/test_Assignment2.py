import sys
import os
sys.path.append("..")

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import joblib 
answers = joblib.load(str(DIR)+"/answers_Assignment2.joblib")

# Import the student solutions
import py487

import numpy as np
np.random.seed(1)
c = np.random.rand(6,4)

import torch

def test_exercise_1():
  A = torch.tensor([[1, 2], [3, 4]])
  B = torch.tensor([[5, 6], [7, 8]])

  C = py487.functional.elementwise_addition(A,B)
  assert torch.all(answers['exercise_1'] == C)

def test_exercise_2():
  A = torch.tensor([[1, 2], [3, 4]])
  B = torch.tensor([[5, 6], [7, 8]])

  C = py487.functional.concatenate_tensors(A,B,0)
  assert torch.all(answers['exercise_2'] == C)

def test_exercise_3():
  A = torch.tensor([1, 2, 3, 4, 5, 6])
  new_shape = (2, 3)
  C = py487.functional.reshape_tensor(A,new_shape)
  assert torch.all(answers['exercise_3'] == C)

def test_exercise_4():
  A = torch.tensor([[1, 2, 3], [4, 5, 6]])
  dim = 1
  C = py487.functional.sum_along_dim(A,dim)
  assert torch.all(answers['exercise_4'] == C)

def test_exercise_5():
  A = torch.tensor([[1, 2], [3, 4]])
  B = torch.tensor([[5, 6], [7, 8]])

  C = py487.functional.matrix_multiply(A,B)
  assert torch.all(answers['exercise_5'] == C)