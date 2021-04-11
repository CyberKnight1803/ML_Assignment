import numpy as np
import pandas as pd

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
