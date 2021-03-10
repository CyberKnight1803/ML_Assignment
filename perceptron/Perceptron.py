from os import get_terminal_size
from tqdm import tqdm

import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

from utils import get_weight

# TODO: Add other Performance Metrix:
#   - Accuracy
#   - Recall
#   - Precision
#   - F-Score
# 
# TODO: Add Decision Boundary 

class Perceptron:
    def __init__(self, num_features, initialization='Random') -> None:
        self.num_features = num_features
        self.w = get_weight((num_features, 1, initialization))

    def classify(self, X):
        y = X @ self.w
        y[y > 0] = 1
        y[y < 0] = -1

        return y

    def get_misclassified(self, X, y):
        y_hat = self.classify(X).squeeze()

        mask = y != y_hat 

        return X[mask], y[mask]

    def sgd(self, X, y, learning_rate=1, epochs=5, print_freq=1):

        accuracy = []

        for i in tqdm(range(epochs)):
            misclassified_X, misclassified_y = self.get_misclassified(X, y)

            acc = 1 - len(misclassified_X) / X.shape[0]
            accuracy.append(acc)

            if acc == 1:
                return accuracy

            if print_freq and i % print_freq == 0:
                print(f"Accuracy at Iteration[{i}]: {acc}")

            self.w += learning_rate * (misclassified_X * misclassified_y.reshape(-1, 1)).sum(axis=0).T.reshape(-1, 1)

        return accuracy

    def test(self, X, y):
        misclassified_X, _ = self.get_misclassified(X, y) # Ignoring Misclassifed y

        acc = 1 - misclassified_X.shape[0] / X.shape[0]
        
        return acc
