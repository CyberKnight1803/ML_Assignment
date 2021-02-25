from os import get_terminal_size
from tqdm import tqdm

import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

from utils import get_weight

class Perceptron:
    def __init__(self, num_features, initialization='Random') -> None:
        self.num_features = num_features
        self.w = get_weight((num_features, 1, initialization))

    def classify(self, x):
        return 1 if self.w.T @ x > 0 else -1

    def get_misclassified(self, X, y):
        misclassified = []
        for x, t in zip(X, y):
            result = t * self.classify(x)
            if result < 0:
                misclassified.append((x, t))

        return misclassified

    def sgd(self, X, y, learning_rate=1, epochs=5, print_freq=1):

        accuracy = []

        for i in tqdm(range(epochs)):
            misclassified = self.get_misclassified(X, y)

            acc = 1 - len(misclassified) / X.shape[0]
            accuracy.append(acc)

            if print_freq and i % print_freq == 0:
                print(f"Accuracy at Iteration[{i}]: {acc}")

            # for x, t in tqdm(misclassified):
            for x, t in misclassified:
            
                self.w += learning_rate * t * x.reshape(-1, 1)

        return accuracy

    def test(self, X, y):
        misclassified = self.get_misclassified(X, y)

        acc = 1 - len(misclassified) / X.shape[0]
        
        return acc

