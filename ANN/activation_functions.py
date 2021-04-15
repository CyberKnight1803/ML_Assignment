# All Activation functions required for deep learning
import numpy as np

class Sigmoid():
    def __call__(self, z):
        S = 1 / (1 + np.exp(-z))
        return S
    
    def derivative(self, z):
        dS = self.__call__(z) * (1 - self.__call__(z))
        return dS

class TanH():
    def __call__(self, z):
        T = np.tanh(z)
        return T
    
    def derivative(self, z):
        dT = 1 - np.power(self.__call__(z), 2)
        return dT

class ReLu():
    def __call__(self, z):
        R = np.maximum(0, z)
        return R
    
    def derivative(self, z):
        dR = np.where(z > 0, 1, 0)
        return dR

class LeakyReLu():
    def __init__(self, grad = 0.01):
        self.grad = grad
    
    def __call__(self, z):
        L = np.where(z > 0, z, self.grad * z)
        return L
    
    def derivative(self, z):
        dL = np.where(z > 0, 1, self.grad)
        return dL

class SoftMax():

    def __call__(self, z, stability=True):
        # Numerical Stability
        if stability:
            self.D = np.max(z, axis=0)
            shift = z - self.D
            t = np.exp(shift)
            S = t / np.sum(t, axis=0)
            return S

        else:
            t = np.exp(z)
            S = t / np.sum(t, axis=0)
            return S

    def derivative(self, z):
        pass

activations = {
    'Sigmoid': Sigmoid,
    'TanH': TanH,
    'ReLu': ReLu,
    'LeakyReLu': LeakyReLu,
    'SoftMax': SoftMax,
}