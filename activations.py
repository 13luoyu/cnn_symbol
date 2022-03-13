"""
激活函数的定义和边界计算
"""

import numpy as np


def relu(x):
    return max(x, 0)

def sigmoid(x):
    """sigmoid函数"""
    return 1.0 / (1.0 + np.exp(-x))



def tanh(x):
    return np.tanh(x)


def atan(x):
    return np.arctan(x)
