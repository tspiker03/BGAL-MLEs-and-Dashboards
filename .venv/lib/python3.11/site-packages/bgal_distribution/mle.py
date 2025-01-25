import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statistics import mean
import random as rand


def delta_MLE(X, y):
    sum_ratio = 0
    for i in range(len(X)):
        sum_ratio += y[i] / X[i]
    delta_hat = (np.mean(y) - (np.mean(X) / len(X)) * sum_ratio) / ((np.mean(X) / (len(X)) * sum(-1 / X)) + 1)
    return delta_hat


def w(X, y, delta):
    # the function w evaluates the terms that contain delta.  This function will be passed into the MLE for sigma
    n = len(X)

    y_minus_delta = []
    for j in range(len(y)):
        y_minus_delta.append(y[j] - delta)
    y_minus_delta_sum = sum(y_minus_delta)
    y_minus_delta_squared = []
    for j in range(len(y)):
        y_minus_delta_squared.append(((y[j] - delta) ** 2) / X[j])
    y_minus_delta_squared_sum = sum(y_minus_delta_squared)
    w_d = (2 * (np.mean(y) - delta) / (n * np.mean(X)) * y_minus_delta_sum) - ((1 / n) * y_minus_delta_squared_sum) \
          - ((1 / n) * ((np.mean(y) - delta) / (np.mean(X))) ** 2 * np.sum(X))
    return w_d


def r_MLE(w_d):
    # returns the MLE for sigma
    return np.sqrt(-w_d)


def mu_MLE(X, y, delta):
    # returns the MLE for mu
    return ((np.mean(y) - delta) / np.mean(X))
