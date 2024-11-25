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

import json 
answers = json.loads(open(str(DIR)+"/answers_Assignment6.json").read())

import pandas as pd
import numpy as np

glove = pd.read_csv('/tmp/glove/glove.6B.100d.txt', sep=" ", quoting=3, header=None, index_col=0)
glove_embedding = {key: val.values for key, val in glove.T.items()}
sentence1 = "The king and the queen had a son who is the prince"
words1 = sentence1.split()
sentence2 = "The king of england is Charles"
words2 = sentence2.split()
n_tokens = max(len(words1),len(words2))

d_k = len(glove_embedding["queen"])
num_heads = 5

V1 = []
for i in range(n_tokens):
    if i < len(words1):
        V1.append(glove_embedding[words1[i].lower()].tolist()*num_heads)
    else:
        V1.append(np.zeros((d_k*num_heads,)))

V2 = []
for i in range(n_tokens):
    if i < len(words2):
        V2.append(glove_embedding[words2[i].lower()].tolist()*num_heads)
    else:
        V2.append(np.zeros((d_k*num_heads,)))
    
V = torch.tensor(np.array([V1,V2]),dtype=torch.float32)

def test_1():
    attention = py487.attention.MultiHeadAttention(d_k*num_heads,num_heads)
    solution = attention.split_heads(V,list_output=True)

    answer = np.array(answers['question_1'])

    assert np.all(np.abs(solution-answer) <= 0.0001)
    
def test_2():
    attention = py487.attention.MultiHeadAttention(d_k*num_heads,num_heads)
    solution = attention.combine_heads(torch.tensor(attention.split_heads(V,list_output=True),dtype=torch.float32),list_output=True)
    answer = np.array(answers['question_2'])

    assert np.all(np.abs(solution-answer) <= 0.0001)
    
def test_3():
    attention = py487.attention.MultiHeadAttention(d_k*num_heads,num_heads)
    solution = attention.scaled_dot_product_attention(V,V,V,list_output=True)
    answer = np.array(answers['question_3'])

    assert np.all(np.abs(solution-answer) <= 0.0001)
    
def test_4():
    attention = py487.attention.MultiHeadAttention(d_k*num_heads,num_heads)
    solution = np.array(attention.forward(V,V,V,list_output=True))
    answer = np.array(answers['question_4'])

    assert np.all(np.abs(solution-answer) <= 0.0001)
    