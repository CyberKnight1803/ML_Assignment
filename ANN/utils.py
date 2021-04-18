import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from csv import writer

def STATS(hyperparameters, layer_dims, DNN, X_train, X_test, y_train, y_test):
    costs = []
    columns = ["lr", "epochs", "bacth_size", "Activation", "Initializer", "Train acc", "Test acc"]
    df = pd.DataFrame(columns=columns)

    for epochs in hyperparameters["epochs"]:
        for batch in hyperparameters["batch_size"]:
            for activation in hyperparameters["activations"]:
                for lr in hyperparameters["lr"]:
        
                    if activation == 'ReLu':
                        Initializers = ["Random", "He"]
                        for initializer in Initializers:
                            model = DNN(layer_dims, lRate=lr, epochs=epochs, activation=activation, initializer=initializer, GD_type='MiniBatchGD', batch_size=batch)
                            costs.append(model.fit(X_train, y_train))
                            train_acc = model.accuracy(X_train, y_train)
                            test_acc = model.accuracy(X_test, y_test)
                            df.loc[-1] = [lr, epochs, batch, activation, initializer, train_acc, test_acc]
                            df.index += 1

                    
                    if activation == 'TanH':
                        Initializers = ["Random", "Xavier", "NormalizedXavier"]
                        for initializer in Initializers:
                            model = DNN(layer_dims, lRate=lr, epochs=epochs, activation=activation, initializer=initializer, GD_type='MiniBatchGD', batch_size=batch)
                            costs.append(model.fit(X_train, y_train))
                            
                            train_acc = model.accuracy(X_train, y_train)
                            test_acc = model.accuracy(X_test, y_test)
                            df.loc[-1] = [lr, epochs, batch, activation, initializer, train_acc, test_acc]
                            df.index += 1
    df = df.sort_index()
    df.to_clipboard()

    print('Resuls\n')
    print(df)

def SAVE(ARCHITECTURE, train_acc, test_acc, fileName):
    columns = ['layer_dims', 'lRate', 'epochs', 'activation', 'initializer', 'GD_type', 'batch_size', 'optimizer', 'optimizer_const', 'train_acc', 'test_acc']
    with open(f'{fileName}.csv', 'a') as infile:
        data =  list(ARCHITECTURE.values()) + [train_acc] + [test_acc]
        writer_object = writer(infile)
        writer_object.writerow(data)
        infile.close()
