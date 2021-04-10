from operator import le
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
        return f"Weights: {self.W}\nBias: {self.b}"

    def propagate(self, X):
        """
        return computed probabilities
        """
        Z = self.W.T @ X + self.b
        A = sigmoid(Z)

        return A

    def fit(self, X, Y, learning_rate=0.75, epochs=1000, print_cost=50, optimizer='BGD'):
        '''
        Optimize the model with the given train and test set using the given optimizers
        Apply the algorithm num_epoch times
        if print_cost, record the cost at the specified intervals
        '''
        if (optimizer == 'SGD'):
            return self.sgd(X, Y)

        costs = []
        accuracies = []

        for i in tqdm(range(epochs)):
            A = self.propagate(X)
            N = X.shape[1]

            # Calculate the Gradients
            dZ = A - Y
            dW = (X @ dZ.T) / N
            db = np.sum(dZ) / N

            # Make Adjustments
            self.W -= learning_rate * dW
            self.b -= learning_rate * db

            if print_cost and i % print_cost:
                costs.append(cross_entropy(Y, A))
                accuracies.append(accuracy(Y, A))

                print(f"[{i} Cost: {costs[-1]} Accuracy: {accuracies[-1]}")

        return costs, accuracies

    def sgd(self, X, Y, learning_rate=0.75, epochs=5, print_cost=50):
        '''
        Optimize the model with the given train and test set using SGD
        Apply the algorithm num_epoch times
        if print_cost, record the cost at the specified intervals
        '''
        costs = []
        accuracies = []

        iterations = 0

        for i in tqdm(range(epochs)):
            for j in range(X.shape[1]):

                x = X[:, j]
                y = Y[:, j]

                A = self.propagate(X).reshape(-1, 1)
                a = self.propagate(x).reshape(-1, 1)

                dz = a - y
                dW = (x * dz).reshape(-1, 1)
                db = np.sum(dz)

                self.W -= learning_rate * dW
                self.b -= learning_rate * db

                if print_cost and iterations % print_cost:
                    costs.append(cross_entropy(y, a))
                    accuracies.append(accuracy(Y, A))

                    print(f"[{iterations} Cost: {costs[-1]} Accuracy: {accuracies[-1]}")

                iterations += 1

        return costs, accuracies

    def classify(self, X):
        '''
        Classify points
        '''
        A = self.propagate(X)

        A[A < 0.5]  = 0
        A[A >= 0.5] = 1

        return A

    def test(self, X, Y):
        '''
        Test the accuracy of the model on the specified test set
        '''
        pass
