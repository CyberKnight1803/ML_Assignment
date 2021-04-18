'''
Requirements:

1. Implement Log Reg for binary classification
2. Use appropriate lr
3. Plot loss and accuracy for model every 50 iterations
4. 10 independent 70:30 splits on given data train model and report avg loss and accuracy over 10 splits
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_dataset, train_test_split
from LR import Model


def test_sgd(learning_rates):
    X, y = load_dataset("./dataset_LR.csv")
    training, testing = train_test_split(X, y)

    print(training["X"].shape, training["y"].shape,
          testing["X"].shape, testing["y"].shape)

    costs = {}
    accuracies = {}

    for lr in learning_rates:
        model = Model(X.shape[1], 'Random')
        cost, accuracy = model.sgd(training["X"].T, training["y"].T, lr=lr)

        costs[lr] = cost
        accuracies[lr] = accuracy

    return costs, accuracies


def test_bgd(learning_rates):
    X, y = load_dataset("./dataset_LR.csv")
    training, testing = train_test_split(X, y)

    print(training["X"].shape, training["y"].shape,
          testing["X"].shape, testing["y"].shape)

    costs = {}
    accuracies = {}

    for lr in learning_rates:
        model = Model(X.shape[1], 'Random')
        cost, accuracy = model.bgd(training["X"].T, training["y"].T, lr=lr)

        costs[lr] = cost
        accuracies[lr] = accuracy

    return costs, accuracies


def evaluate():
    X, y = load_dataset("./dataset_LR.csv")

    performance = pd.DataFrame(columns=[
        "lr", "Test Sequence", "Epochs", "Accuracy", "Precision", "Recall", "F1-Score"
    ])

    accuracy = []

    for i in range(10):
        training, testing = train_test_split(X, y)

        model = Model(X.shape[1])

        model.bgd(training["X"].T, training["y"].T)
        # model.sgd(training["X"].T, training["y"].T)

        metrics = model.test(testing["X"].T, testing["y"].T)
        accuracy.append(metrics["accuracy"])

        performance.loc[-1] = [lr, i, metrics["epochs"], metrics["accuracy"],
                               metrics["precision"], metrics["recall"], metrics["f1-score"]]
        performance.index += 1

    performance = performance.sort_index()
    performance.to_clipboard()

    print(performance)
    print(f"Mean Accuracy over 10 tests: {performance['Accuracy'].mean()}")
    print(f"Mean Precision over 10 tests: {performance['Precision'].mean()}")
    print(f"Mean Recall over 10 tests: {performance['Recall'].mean()}")
    print(f"Mean F1-Score over 10 tests: {performance['F1-Score'].mean()}")


def make_plots():
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    # learning_rates = [0.0001, 0.001, 0.01]

    costs_sgd, accuracies_sgd = test_sgd(learning_rates)
    costs_bgd, accuracies_bgd = test_bgd(learning_rates)

    figure, axes = plt.subplots(2, 2)

    for lr, cost in costs_sgd.items():
        axes[0, 0].plot(cost)

    axes[0, 0].legend(costs_sgd.keys())
    axes[0, 0].set_title("Cost vs Iteration (SGD)")

    for lr, cost in costs_bgd.items():
        axes[1, 0].plot(cost)

    axes[1, 0].legend(costs_bgd.keys())
    axes[1, 0].set_title("Cost vs Epoch (BGD)")

    for lr, acc in accuracies_sgd.items():
        axes[0, 1].plot(acc)

    axes[0, 1].legend(accuracies_sgd.keys())
    axes[0, 1].set_title("Accuracy vs Iteration (SGD)")

    for lr, acc in accuracies_bgd.items():
        axes[1, 1].plot(acc)

    axes[1, 1].legend(accuracies_bgd.keys())
    axes[1, 1].set_title("Accuracy vs Epoch (BGD)")

    figure.tight_layout()
    plt.show()


def analysis():
    learning_rates = [0.001, 0.0025, 0.005]

    costs_sgd, accuracies_sgd = test_sgd(learning_rates)
    costs_bgd, accuracies_bgd = test_bgd(learning_rates)

    figure, axes = plt.subplots(2, len(learning_rates))

    for i, lr in enumerate(learning_rates):
        axes[0, i].plot(costs_sgd[lr])
        axes[0, i].plot(accuracies_sgd[lr])
        axes[0, i].legend(['costs', 'accuracy'])
        axes[0, i].set_title(f"lr = {lr} (SGD)")

    for i, lr in enumerate(learning_rates):
        axes[1, i].plot(costs_bgd[lr])
        axes[1, i].plot(accuracies_bgd[lr])
        axes[1, i].legend(['costs', 'accuracy'])
        axes[1, i].set_title(f"lr = {lr} (BGD)")



    figure.tight_layout()
    plt.show()




if __name__ == "__main__":
    # make_plots()

    # evaluate()

    analysis()