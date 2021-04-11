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


def test_sgd():
    X, y = load_dataset("./dataset_LR.csv")
    training, testing = train_test_split(X, y)

    model = Model(X.shape[1], 'Zeros')

    lr = 0.005

    loss, accuracy = model.sgd(training["X"].T, training["y"].T, lr=lr)

    print(f"SGD: \nFirst 10 Loss: \n{loss[:10]}")
    print(f"First 10 accuracy: \n{accuracy[:10]}")

    plt.plot(loss)
    plt.plot(accuracy)

    plt.legend(["loss", "accuracy"])

    plt.show()

    accuracy = model.test(testing["X"].T, testing["y"].T)
    print(f"Final Accuracy on test set: {accuracy}")


def test_bgd():
    X, y = load_dataset("./dataset_LR.csv")
    training, testing = train_test_split(X, y)

    model = Model(X.shape[1], 'Zeros')

    lr = 0.005

    loss, accuracy = model.bgd(training["X"].T, training["y"].T, lr=lr)

    print(f"SGD: \nFirst 10 Loss: \n{loss[:10]}")
    print(f"First 10 accuracy: \n{accuracy[:10]}")

    plt.plot(loss)
    plt.plot(accuracy)

    plt.legend(["loss", "accuracy"])

    plt.show()

    accuracy = model.test(testing["X"].T, testing["y"].T)
    print(f"Final Accuracy on test set: {accuracy}")


def evaluate():
    X, y = load_dataset("./dataset_LR.csv")

    performance = pd.DataFrame(columns=[
        "lr", "Test Sequence", "Epochs", "Accuracy"
    ])

    accuracy = []

    lr = 0.75 # 0.01
    epochs = 500 # 50

    for i in range(10):
        training, testing = train_test_split(X, y)

        model = Model(X.shape[1])

        model.bgd(training["X"].T, training["y"].T, lr=lr,epochs=epochs)
        # model.sgd(training["X"].T, training["y"].T, lr=lr, epochs=epochs)

        acc = model.test(testing["X"].T, testing["y"].T)
        accuracy.append(acc)

        performance.loc[-1] = [lr, i, epochs, acc]
        performance.index += 1


    performance = performance.sort_index()
    performance.to_clipboard()

    print(performance)
    print(f"Mean Accuracy over 10 tests: {np.mean(accuracy)}")


if __name__ == "__main__":
    evaluate()
