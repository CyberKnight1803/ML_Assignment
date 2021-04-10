'''
Requirements:

1. Implement Log Reg for binary classification
2. Use appropriate lr
3. PLot loss and accuracy for model every 50 iterations
4. 10 independent 70:30 splits on given data train model and report avg loss and accuracy over 10 splits
'''

import pandas as pd
import matplotlib.pyplot as plt
from utils import load_dataset
from LR import Model


def main():
    X, y = load_dataset("./dataset_LR.csv")

    performance = pd.DataFrame(columns=[
        "lr", "Test Sequence", "Epochs", "Accuracy"
    ])

    lr = 0.75

    model = Model(X.shape[1])
    loss, accuracy = model.fit(X.T, y.T, lr, optimizer='SGD')

    plt.plot(loss)
    plt.plot(accuracy)

    plt.legend(["loss", "accuracy"])

    plt.show()


if __name__ == "__main__":
    main()
