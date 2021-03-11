import matplotlib.pyplot as plt

import pandas as pd

from Perceptron import Perceptron
from utils import load_dataset, train_test_split

MAX_EPOCHS = int(1.5e5)


def find_best_lr(training, testing):
    learning_rates = [0.125, 0.25, 0.45, 0.5, 0.75, 0.85, 0.9, 0.99, 1]

    accuracies = {}

    df = pd.DataFrame(
        columns=["lr", "Iterations", "Accuracy", "Precision", "Recall", "F1-Score"])

    for lr in learning_rates:
        perceptron = Perceptron(X.shape[1])
        accuracy = perceptron.fit(
            training["X"], training["y"], learning_rate=0.45, epochs=MAX_EPOCHS, print_freq=0)
        accuracies[lr] = accuracy

        metrics = perceptron.test(testing["X"], testing["y"])

        df.loc[-1] = [lr, metrics["iterations"], metrics['accuracy'],
                      metrics['precision'],
                      metrics['recall'],
                      metrics['f1-score']]

        df.index += 1

    df = df.sort_index()
    df.to_clipboard()

    print("\nResults...")
    print(df)

    for lr, accuracy in accuracies.items():
        plt.plot(accuracy)

    plt.legend(learning_rates)
    plt.show()


def ez(training, testing):
    learning_rate = 0.45
    perceptron = Perceptron(X.shape[1])
    accuracy = perceptron.fit(
        training["X"], training["y"], learning_rate=learning_rate, epochs=MAX_EPOCHS, print_freq=0)

    plt.plot(accuracy)
    plt.show()

    df = pd.DataFrame(
        columns=["lr", "Accuracy", "Precision", "Recall", "F1-Score"])

    metrics = perceptron.test(testing["X"], testing["y"])

    df.loc[-1] = [learning_rate, metrics['accuracy'],
                  metrics['precision'],
                  metrics['recall'],
                  metrics['f1-score']]

    df.index += 1

    print(df)


if __name__ == '__main__':
    # path = 'dataset_LP_1.txt'
    path = 'dataset_LP_2.csv'

    X, y = load_dataset(path)

    # Sanity Checks
    print(X.shape, y.shape)

    training, testing = train_test_split(X, y)

    # Sanity Checks
    print(training["X"].shape, training["y"].shape,
          testing["X"].shape, testing["y"].shape)

    find_best_lr(training, testing)
