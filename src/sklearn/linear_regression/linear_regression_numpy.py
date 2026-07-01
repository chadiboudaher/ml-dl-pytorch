import numpy as np

"""
Linear Regression from scratch (Numpy + Gradient Descent)
"""

class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 1000):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.w = None
        self.b = 0.0
        self.loss_history = []

    