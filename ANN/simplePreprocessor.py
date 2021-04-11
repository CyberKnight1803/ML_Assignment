import numpy as np
import pandas as pd

class SimplePreprocessor():

    def OneHot(self, y):
        K = y.max()
        N = len(y)
        Y = y - 1
        H = np.zeros((K, N))
        H[Y, np.arange(N)] = 1
        return H

    def Standardize(self, X, att_lst):
        mean = X.loc[:][att_lst].mean()
        std = X.loc[:][att_lst].std()

        X.loc[:][att_lst] = (X.loc[:][att_lst] - mean) / std
        return X

    def train_test_split(self, X, y, test_size=0.30):
        N = X.shape[1]
        trainSize = int(N * (1 - test_size))
        
        X_train = X[:, :trainSize]
        X_test = X[:, trainSize:]
        y_train = y[:, :trainSize]
        y_test = y[:, trainSize:]

        return X_train, X_test, y_train, y_test
