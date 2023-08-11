# ML for Carbon Sequestration

Code for the manuscript "A Comparison between Machine Learning Methods for Carbon Sequestration Estimation based on Remote Sensing Data". <br>
Mehdi Fasihi, Beatrice Portelli, Luca Cadez, Antonio Tomao, Giorgio Alberti, Giuseppe Serra.


## Requirements

For the libraries needed to run the code check `requirements.txt`

You can install the needed libraries using `pip install -r requirements.txt`

## Files overview

- **`results`** <br>
  This folder contains the outputs of the models described in the paper. <br>
  Be careful, running some of the code in this repository might overwrite the results folder. Please rename the folder to keep the original results. <br>
  Also note that the training features (X_train and X_test, as well as y_train) present in the `predictions--*` files have been removed and only the target values/predictions (y_test and y_pred) needed to reproduce the figures in the paper have been kept. Training data will be made available upon request.
  
- **`py_scripts`** <br>
  This folder contains the *.py version of all the *.ipynb notebooks described below.

- **`configs.py`** <br>
  Contains constants such as:
  - names of the target variables (denoted as `TARGET_*`)
  - definitions of the input configurations (denoted as `CONFIG_*`)

- **`experiments_to_run.py`** <br>
  Contains the list of models, hyperparameters to tune and target variables that we are interested in

- **`00_run_all_experiments.ipynb`** <br>
  Contains the code to run the training and evaluation of the models and configurations contained in `experiments_to_run.py`. The code does the following for each model:
  - load `INFC_2015_climatic.csv` (data) and select the relevant subset of input features (according to the CONFIG) and target (according to TARGET)
  - perform a randomized grid search to find the best hyperparameters for the model
  - train and evaluate the model 10 times with different folds of train/test data (66/33)
  - calculate the following metrics on all the folds: MSE, R2, MAPE, RMSE, NRMSE
  - save the predictions and metrics in `results/predictions--TARGET--CONFIG--MODEL.pickle` and `results/metrics--TARGET--CONFIG--MODEL.pickle`

- **`01_summarize_all_metrics.ipynb`** <br>
  Saves the hyperparameters of all the best models in `results/hyperparams--TARGET.[md/xlsx]`

- **`02_create_results_table.ipynb`** <br>
  Loads all metrics file available in the `results` folder, computes the average metrics (and std). Saves the resulting table(s) in `results/table_TARGET.[md/xlsx]`

- **`03_feature_importance.ipynb`** <br>
  Computes the feature importances and creates the feature importance plots. Saves the images in `results/feature_importance_MODEL_TARGET.[pdf/png]` and `results/feature_importance_MODEL_TARGET_rot90.png` <br>
  Note: running this code on the results folder shared with the repository will generate a warning for some models. This is normal since we had to remove the input features needed to calculate the feature importances of some kinds of models.

- **`04_plot_trends.ipynb`** <br>
  Loads all the predictions, plots the correlation between real and predicted values and their marginal distributions. Saves the results in `results/trends_with_margins_MODEL_TARGET.[pdf/png]`
