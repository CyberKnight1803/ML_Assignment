import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simplePreprocessor import SimplePreprocessor
from deepNN import DNN
from utils import STATS

def main():
    preprocessor = SimplePreprocessor()
    X, y = preprocessor.load_dataset('./dataset_NN.csv')
    
    y = preprocessor.OneHot(y)

    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = preprocessor.train_test_split(X.T, y)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    architecture = [X_train.shape[0], 16, y_train.shape[0]]

    model = DNN(architecture, lRate=0.1, epochs=500, activation='ReLu', initializer='He', GD_type='MiniBatchGD', batch_size=64)
    accs = model.fit(X_train, y_train)

    print("Training Completed...")

    plt.plot(accs)

    test_acc = model.accuracy(X_test, y_test)
    print(f"Test Accuracy: {test_acc}")

if __name__ == "__main__":
    main()