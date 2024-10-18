import torch

import numpy as np
import copy

def minimize_gradient_descent(gradient_funcs,alpha,theta0,tol=1e-10,max_iter=500):
    """
    You can stop optimizing when the max absolute change in theta is < tol.
    You can also stop when you reach max_iterations.
    Make sure you use the theta from the previous iteration for all your calculations until the next "epoch". In other words, make sure you are making copies correctly and not updated say w1 and then using it to help you update w2. The updates to the parameters should be done in parallel.
    """
    thetas = [theta0]
    return thetas

def minimize_gradient_descent_analytically(J_func,alpha,theta0,h,tol=1e-10,max_iter=500,debug=True):
    """
    You can stop optimizing when the max absolute change in theta is < tol.
    You can also stop when you reach max_iterations.
    Make sure you use the theta from the previous iteration for all your calculations until the next "epoch". In other words, make sure you are making copies correctly and not updated say w1 and then using it to help you update w2. The updates to the parameters should be done in parallel.
    """
    thetas = [theta0]
    return thetas

# this is a test
