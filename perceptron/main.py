import matplotlib.pyplot as plt

from Perceptron import Perceptron
from utils import load_dataset, train_test_split

def find_best_lr(training, testing):
    learning_rates = [0.125, 0.25, 0.45, 0.5, 0.75, 0.85, 0.9, 0.99, 1]

    accuracies = {}

    for lr in learning_rates:
        perceptron = Perceptron(X.shape[1])
        accuracy = perceptron.sgd(training["X"], training["y"], learning_rate=0.45, epochs=int(1e6), print_freq=0)
        accuracies[lr] = accuracy

        testing_acc = perceptron.test(testing["X"], testing["y"])
        print(f"[lr: {lr}] Tesiting Set Accuracy: {testing_acc}")

    
    for lr, accuracy in accuracies.items():
        plt.plot(accuracy)
        
        
    plt.legend(learning_rates)
    plt.show()


def ez(training, testing):
    perceptron = Perceptron(X.shape[1])
    accuracy = perceptron.sgd(
        training["X"], training["y"], learning_rate=0.45, epochs=100, print_freq=0)   

    plt.plot(accuracy)
    plt.show()

    testing_acc = perceptron.test(testing["X"], testing["y"])
    print(f"Accuracy on Testing Set: {testing_acc}")

if __name__ == '__main__':
    path = 'dataset_LP_1.txt'
    # path = 'dataset_LP_2.csv'

    X, y = load_dataset(path)

    # Sanity Checks
    print(X.shape, y.shape)

    training, testing = train_test_split(X, y)

    # Sanity Checks
    print(training["X"].shape, training["y"].shape,
          testing["X"].shape, testing["y"].shape)

    ez(training, testing)