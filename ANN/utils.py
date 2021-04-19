import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from csv import writer

def SAVE(ARCHITECTURE, train_acc, test_acc, fileName):
    columns = ['layer_dims', 'lRate', 'epochs', 'activation', 'initializer', 'GD_type', 'batch_size', 'optimizer', 'optimizer_const', 'train_acc', 'test_acc']
    with open(f'{fileName}.csv', 'a') as infile:
        data =  list(ARCHITECTURE.values()) + [train_acc] + [test_acc]
        writer_object = writer(infile)
        writer_object.writerow(data)
        infile.close()
