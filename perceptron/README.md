# Linear Perceptron Model

## Authors: 
- [Omkar Pitale (2019A7PS0083H)](https://github.com/CyberKnight1803)
- [Aditya Chopra (2019A7PS0178H)](https://github.com/adichopra11)
- [Anuradha Pandey (2019A7PS0265H)](https://github.com/pandeyanuradha)


## [About the Model](Perceptron_Report.pdf)

## Using the Model

1. Create a conda environment and install all dependencies

```sh
# Create environment and Install Dependencies
conda create -n ml python=3.9 jupyter numpy pandas matplotlib tqdm 

# Activate Environment
conda activate ml
```

2. Using the Jupyter Notebook
   
   - Start a Jupyter Notebook Instance and open [Perceptron.ipynb](Perceptron.ipynb) in the Jupyter File explorer
```sh
jupyter-notebook
``` 
  
3.  Using the cli
    - Open [main.py](main.py) in Text Editor of Choice
    - Select the dataset to test the model on. 
        - For example: To test the Model on [Dataset 2](dataset_LP_2.csv)
            ```py
            # path = 'dataset_LP_1.txt'
            path = 'dataset_LP_2.csv'
            ```
    - Select the Mode to run in: 
      - Training and Testing on a fixed Learning Rate
      - Analysis over various Learning Rates
      - Plotting Iteration vs Accuracy Curve for both Datasets (selecting path not required for this)
      - For example: To run a Learning Rate Analysis
        ```py
        # batch=False To use stochastic Gradient Descent instead of Batch Gradient Descent
        find_best_lr(path, batch=False) 
        # ez(path, learning_rate=0.01, batch=False)
        # plot_training_cruves()
        ```
    - Run the [main.py](main.py) file
        ```sh
        python main.py
        ```

