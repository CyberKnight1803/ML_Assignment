from tqdm import tqdm
from collections import Counter
import random

from utils import get_weight

# TODO: Add checks for division by zero

MAX_ITERATIONS = int(1e6)


class Perceptron:
    def __init__(self, num_features, initialization='Random') -> None:
        self.w = get_weight((num_features, 1, initialization))
        self.iterations = 0

    def classify(self, X):
        y = X @ self.w
        y[y > 0] = 1
        y[y < 0] = -1

        return y

    def get_misclassified(self, X, y):
        y_hat = self.classify(X).squeeze()

        mask = y != y_hat

        return X[mask], y[mask]

    def fit(self, X, y, learning_rate=1, epochs=5, print_freq=1, batch=True):

        if not batch:
            return self.sgd(X, y, learning_rate=learning_rate, iterations=epochs * X.shape[0], print_freq=print_freq)

        accuracy = []

        for i in tqdm(range(epochs)):
            misclassified_X, misclassified_y = self.get_misclassified(X, y)

            acc = 1 - misclassified_X.shape[0] / X.shape[0]
            accuracy.append(acc)

            self.iterations += misclassified_X.shape[0]

            if (self.iterations > MAX_ITERATIONS):
                diff = self.iterations - MAX_ITERATIONS

                misclassified_X = misclassified_X[:-diff]
                misclassified_y = misclassified_y[:-diff]

            if acc == 1:
                return accuracy

            if print_freq and i % print_freq == 0:
                print(f"Accuracy at Iteration[{i}]: {acc}")

            self.w += learning_rate * \
                (misclassified_X * misclassified_y.reshape(-1, 1)
                 ).sum(axis=0).T.reshape(-1, 1)

            if (self.iterations > MAX_ITERATIONS):
                self.iterations -= (self.iterations - MAX_ITERATIONS)
                break

        print(f"Trained for {self.iterations} iterations")
        return accuracy

    def sgd(self, X, y, learning_rate=1, iterations=1000000, print_freq=0):

        accuracy = []
        for i in tqdm(range(iterations)):
            misclassified_X, misclassified_y = self.get_misclassified(X, y)

            acc = 1 - misclassified_X.shape[0] / X.shape[0]
            accuracy.append(acc)

            if acc == 1:
                return accuracy

            if print_freq and i % print_freq == 0:
                print(f"Accuracy at Iteration[{i}]: {acc}")

            idx = random.randrange(0, misclassified_X.shape[0])
            self.w += (learning_rate *
                       misclassified_X[idx] * misclassified_y[idx]).reshape(-1, 1)

        self.iterations = iterations

        print(f"Trained for {self.iterations} iterations")
        return accuracy

    def test(self, X, y):
        y_hat = self.classify(X).squeeze()

        y = y.tolist()
        y_hat = y_hat.tolist()

        counts = Counter(zip(y_hat, y))

        tp = counts[1, 1]
        fp = counts[1, -1]
        tn = counts[-1, -1]
        fn = counts[-1, 1]

        metrics = {}

        metrics["iterations"] = self.iterations
        metrics["accuracy"] = (tp + tn) / X.shape[0]
        metrics["recall"] = tp / (tp + fn)
        metrics["precision"] = tp / (tp + fp)
        metrics["f1-score"] = (2 * tp) / (2 * tp + fp + fn)

        return metrics
