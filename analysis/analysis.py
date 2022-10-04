import numpy as np


def mean_squared_error(y_predicted, y_actual):
    return ((y_predicted - y_actual) ** 2).mean(axis=1)
