import numpy as np
import pandas as pd

class SimplePreprocessor():

    def load_dataset(self, path):
        df = pd.read_csv(path).sample(frac=1)

        X = df[df.columns[:-1]].to_numpy()
        y = df[df.columns[-1]].to_numpy().reshape(-1, 1)

        return X, y

    def OneHot(self, y):
        K = y.max()
        N = len(y)
        Y = y - 1
        H = np.zeros((K, N))
        H[Y, np.arange(N)] = 1
        return H

    def StandardizeDF(self, X, att_lst):
        mean = X.loc[:][att_lst].mean()
        std = X.loc[:][att_lst].std()

        X.loc[:][att_lst] = (X.loc[:][att_lst] - mean) / std
        return X

    def Standardize(self, X_train, X_test):
        mean = np.mean(X_train, axis=1, keepdims=True)
        std = np.std(X_train, axis=1, keepdims=True)

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        return X_train, X_test
    
    def Normalize(self, X_train, X_test):
        max = np.max(X_train, axis=1, keepdims=True)
        min = np.min(X_train, axis=1, keepdims=True)

        X_train = (X_train - min) / (max - min)
        X_test = (X_test - min) / (max - min)
        return X_train, X_test

    def train_test_split(self, X, y, test_size=0.30, scaling='Normalize'):
        N = X.shape[1]
        trainSize = int(N * (1 - test_size))
        
        X_train = X[:, :trainSize]
        X_test = X[:, trainSize:]
        y_train = y[:, :trainSize]
        y_test = y[:, trainSize:]

        if scaling == 'Normalize':
            X_train, X_test = self.Normalize(X_train, X_test)
        else:
            X_train, X_test = self.Standardize(X_train, X_test)
        return X_train, X_test, y_train, y_test
