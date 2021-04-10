import numpy as np
import pandas as pd


def get_weight(shape, initialization='Random'):
    if initialization == 'Random':
        return np.random.randn(shape[0], shape[1])

    elif initialization == 'Zeros':
        return np.zeros(shape)

    elif initialization == 'Ones':
        return np.ones(shape)



def load_dataset(path, header=None):
    df = pd.read_csv(path).sample(frac=1)

    X = df[df.columns[:-1]].to_numpy()
    y = df[df.columns[-1]].to_numpy().reshape(-1, 1)

    return X, y


def train_test_split(X, y):
    assert X.shape[0] == y.shape[0]

    length = X.shape[0]

    training_size = int(length * 0.7)

    training = {}
    training["X"] = X[:training_size]
    training["y"] = y[:training_size]

    testing = {}
    testing["X"] = X[training_size:]
    testing["y"] = y[training_size:]

    return training, testing
