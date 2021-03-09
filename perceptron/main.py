import matplotlib.pyplot as plt

from Perceptron import Perceptron
from utils import load_dataset, train_test_split

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

    perceptron = Perceptron(X.shape[1])
    accuracy = perceptron.sgd(
        training["X"], training["y"], learning_rate=0.5, epochs=100, print_freq=0)

    plt.plot(accuracy)
    plt.show()

    testing_acc = perceptron.test(testing["X"], testing["y"])
    print(f"Accuracy on Testing Set: {testing_acc}")
