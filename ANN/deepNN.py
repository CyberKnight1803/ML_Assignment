import numpy as np
import matplotlib.pyplot as plt
from layers import Layer
from loss_functions import LogLoss
from gradientDescent import GD_variants
from regularizations import regularizers


class DNN():

    def __init__(self, layer_dims, momentum=0, beta=0.99, lRate=0.45, epochs=5000, activation='ReLu', initializer = 'He', GD_type='StochasticGD',
    batch_size=None, optimizer=None, regularizer=None, regularizer_const=0):
        self.lRate = lRate
        self.epochs = epochs
        self.loss = LogLoss()
        self.optimizer = optimizer
        self.acc = []
        self.regularizer = regularizer
        if regularizer != None:
            self.regularizer = regularizers[regularizer](regularizer_const)
        
        self.GD_type = None
        if GD_type == 'BatchGD':
            self.GD_type = GD_variants[GD_type](self.lRate)
        elif GD_type == 'StochasticGD':
            self.GD_type = GD_variants[GD_type](self.lRate, momentum, beta, self.optimizer)
        elif GD_type == 'MiniBatchGD':
            self.GD_type = GD_variants[GD_type](self.lRate, momentum, beta, batch_size, self.optimizer)

        self.layers = []
        self.n_layers = len(layer_dims) - 1

        #Initializing all layers with ReLu except last
        for l in range(1, self.n_layers):
            layer_shape = (layer_dims[l], layer_dims[l - 1])
            self.layers.append(Layer(l, layer_shape, activation, initializer, self.regularizer))
            print(self.layers[l - 1].__str__())
        
        layer_shape = (layer_dims[self.n_layers], layer_dims[self.n_layers - 1]) 
        self.layers.append(Layer(self.n_layers, layer_shape, 'SoftMax', 'Random', self.regularizer))
        print(self.layers[self.n_layers - 1].__str__())

    def forward_propagation(self, X):
        A = X
        caches = []

        for layer in self.layers:
            _A = A
            A, Z = layer.forward_pass(_A)

            caches.append((_A, Z))

        return A, caches

    def compute_cost(self, y, AL):

        J = - np.sum(self.loss(y, AL)) / self.m
        if self.regularizer != None:
            J += self.regularizer(self.layers, self.m)
            
        return J

    def backward_propagation(self, AL, y, caches):
        dAL = self.loss.derivative(y, AL)

        _dA = dAL
        for l in reversed(range(self.n_layers)):
            dA = _dA
            _dA = self.layers[l].backward_pass(dA, caches[l], y, AL)

    def fit(self, X, y, print_cost=False):
        self.m = X.shape[1]
        self.costs = []
        mechanism = {
            'forward_prop' : self.forward_propagation,
            'backward_prop' : self.backward_propagation,
            'compute_cost' : self.compute_cost,
            'accuracy' : self.accuracy,
        }

        for i in range(0, self.epochs):
            self.GD_type(X, y, self.layers, mechanism, self.costs, i, print_cost=print_cost)

        return self.costs, self.acc
    
    def predict(self, X):
        A, _ = self.forward_propagation(X)
        P = np.argmax(A, axis=0)
        return P
    
    def accuracy(self, X, y):
        P = self.predict(X)
        _y = np.argmax(y, axis=0)

        acc = np.mean(P == _y)
        self.acc.append(acc)
        return acc
        
    def costPlot(self):
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('cost')
        plt.xlabel('epochs')
        plt.show()

    def accPlot(self):
        plt.plot(np.squeeze(self.acc))
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.show()
