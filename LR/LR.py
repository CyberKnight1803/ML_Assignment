from typing import Counter
import numpy as np
from tqdm import tqdm
from utils import get_weight


EPSILON = 1e-15


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cross_entropy(Y, A):
    P = np.clip(A, EPSILON, 1 - EPSILON)
    cost = -np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P))

    return cost


def accuracy(Y, A):
    A[A < 0.5] = 0
    A[A >= 0.5] = 1
    return np.mean(Y == A)


class Model:
    def __init__(self, num_features, initialization='Random') -> None:
        self.W = get_weight((num_features, 1), initialization)
        self.b = 0

    def __str__(self) -> str:
        return f"Weights: {self.W.shape}\nBias: {self.b.shape}"

    def propagate(self, X):
        """
        return computed probabilities
        """
        Z = self.W.T @ X + self.b
        A = sigmoid(Z)

        return A

    def bgd(self, X, Y, lr=0.75, epochs=1000, print_cost=1):
        '''
        Optimize the model with the given train and test set using the given optimizers
        Apply the algorithm num_epoch times
        if print_cost, record the cost at the specified intervals
        '''
        costs = []
        accuracies = []

        self.epochs = epochs

        for i in tqdm(range(epochs)):
            A = self.propagate(X)
            N = X.shape[1]

            # Calculate the Gradients
            dZ = A - Y
            dW = (X @ dZ.T) / N
            db = np.sum(dZ) / N

            # Make Adjustments
            self.W -= lr * dW
            self.b -= lr * db

            if print_cost and i % print_cost == 0:
                costs.append(cross_entropy(Y, A))
                accuracies.append(accuracy(Y, A))

                # print(f"[{i} Cost: {costs[-1]} Accuracy: {accuracies[-1]}")

        return costs, accuracies

    def sgd(self, X, Y, lr=0.01, epochs=50, print_cost=50):
        '''
        Optimize the model with the given train and test set using SGD
        Apply the algorithm num_epoch times
        if print_cost, record the cost at the specified intervals
        '''
        costs = []
        accuracies = []

        self.epochs = epochs

        iterations = 0

        for i in tqdm(range(epochs)):
            for j in range(X.shape[1]):

                x = X[:, j].reshape(-1, 1)
                y = Y[:, j].reshape(-1, 1)

                A = self.propagate(X)
                a = self.propagate(x)

                dz = a - y
                dW = (x * dz)
                db = dz

                self.W -= lr * dW
                self.b -= lr * db

                if print_cost and iterations % print_cost == 0:
                    costs.append(cross_entropy(y, a))
                    accuracies.append(accuracy(Y, A))
                    # print(
                    #     f"[{iterations} Cost: {costs[-1]} Accuracy: {accuracies[-1]}")

                iterations += 1

        return costs, accuracies

    def classify(self, Y):
        '''
        Classify points
        '''

        Y[Y < 0.5] = 0
        Y[Y >= 0.5] = 1

        return Y

    def test(self, X, Y):
        '''
        Test the accuracy of the model on the specified test set
        '''
        Y_hat = self.propagate(X)
        P = self.classify(Y_hat).squeeze().tolist()
        Y = Y.squeeze().tolist()

        counts = Counter(zip(P, Y))

        tp = counts[1, 1]
        fp = counts[1, 0]
        tn = counts[0, 0]
        fn = counts[0, 1]

        metrics = {}

        metrics["epochs"] = self.epochs
        metrics["accuracy"] = (tp + tn) / len(Y)
        metrics["precision"] = tp / (tp + fp)
        metrics["recall"] = tp / (tp + fn)
        metrics["f1-score"] = (2 * tp) / (2 * tp + fp + fn)

        return metrics
