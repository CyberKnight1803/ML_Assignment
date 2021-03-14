import matplotlib.pyplot as plt

import pandas as pd

from Perceptron import Perceptron
from utils import load_dataset, train_test_split

MAX_EPOCHS = int(1.5e5)


def find_best_lr(path, batch=True):
    X, y = load_dataset(path)

    # Sanity Checks
    print(X.shape, y.shape)

    training, testing = train_test_split(X, y)

    # Sanity Checks
    print(training["X"].shape, training["y"].shape,
          testing["X"].shape, testing["y"].shape)

    learning_rates = [0.125, 0.25, 0.45, 0.5, 0.75, 0.85, 0.9, 0.99, 1]

    accuracies = {}

    df = pd.DataFrame(
        columns=["lr", "Iterations", "Accuracy", "Precision", "Recall", "F1-Score"])

    for lr in learning_rates:
        perceptron = Perceptron(X.shape[1])
        accuracy = perceptron.fit(
            training["X"], training["y"], learning_rate=0.45, print_freq=0, batch=batch)
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

    return learning_rates, accuracies


def ez(training, testing, learning_rate=1, batch=True):
    perceptron = Perceptron(X.shape[1])
    accuracy = perceptron.fit(
        training["X"], training["y"], learning_rate=learning_rate, epochs=MAX_EPOCHS, print_freq=0, batch=batch)

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


def plot_training_cruves():
    lr_1, acc_1 = find_best_lr('dataset_LP_1.txt', batch=False)
    lr_2, acc_2 = find_best_lr('dataset_LP_2.csv', batch=False)

    figure, axes = plt.subplots(1, 2)

    print(figure, axes)

    for lr, accuracy_1 in acc_1.items():
        axes[0].plot(accuracy_1)

    axes[0].legend(lr_1)
    axes[0].set_title("Dataset 1")

    for lr, accuracy_2 in acc_2.items():
        axes[1].plot(accuracy_2)

    axes[1].legend(lr_2)
    axes[1].set_title("Dataset 2")

    figure.tight_layout()
    plt.show()


if __name__ == '__main__':
    # path = 'dataset_LP_1.txt'
    path = 'dataset_LP_2.csv'

    find_best_lr(path, batch=False)
    # ez(training, testing, learning_rate=0.01, batch=False)
    # plot_training_cruves()
