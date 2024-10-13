import sys
import os
sys.path.append("..")

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import joblib 
answers = joblib.load(str(DIR)+"/answers_Assignment3.joblib")

# Import the student solutions
import py487

import torch
import pandas as pd
import numpy as np

from sklearn import datasets
np.random.seed(4)
X, t_fruit = datasets.make_blobs(n_samples=100, centers=3, n_features=2, center_box=(0, 10))

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X, t_fruit)
y_fruit_probs_pred = pd.DataFrame(clf.predict_proba(X),columns=clf.classes_)
y_fruit_probs = pd.get_dummies(t_fruit).astype(float)

def test_1():
    solution = float(torch.nn.functional.kl_div(torch.tensor(np.log(y_fruit_probs_pred.values+1e-10)), torch.tensor(y_fruit_probs.values),reduction='batchmean').numpy())
    answer = py487.functional.kl_div(np.log(y_fruit_probs_pred.values+1e-10),y_fruit_probs.values)
    assert np.abs(solution-answer) <= 0.0001

def test_2():
    solution = float(torch.nn.functional.cross_entropy(torch.tensor(np.log(y_fruit_probs_pred.values+1e-10)), torch.tensor(y_fruit_probs.values)).numpy())
    answer = py487.functional.cross_entropy(np.log(y_fruit_probs_pred.values+1e-10),y_fruit_probs.values)
    assert np.abs(solution-answer) <= 0.0001