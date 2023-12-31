#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold, RepeatedKFold, LeaveOneOut

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

from tqdm.auto import tqdm

from copy import deepcopy

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def normalized_root_mean_squared_error(y_true, y_pred, norm_factor=None):
    if norm_factor is None:
        assert False, "Set norm_factor (for example the average target value for the training set)"
    rmse = root_mean_squared_error(y_true, y_pred)
    return (rmse / norm_factor)*100

from configs import *
from experiments_to_run import MODELS, CONFIGS, TARGETS


OVERWRITE = False


total_iterations = len(MODELS)*len(CONFIGS)*len(TARGETS)
curr_iteration = 0

for MODEL in MODELS:
    for CONFIG in CONFIGS:
        for TARGET in TARGETS:
            
            curr_iteration+=1
            print(f"---------------- [{curr_iteration} / {total_iterations}]")
            
            if not OVERWRITE:
                conf = f"{TARGET}--{CONFIG.name}--{MODEL[0](_).__class__.__name__}"
                save_path = f"results/metrics--{conf}.pickle"
                if os.path.exists(save_path):
                    print("File already exists. Skipping", conf)
                    continue
                else:
                    print("Running", conf)

            # fixing random seed as soon as possible
            # for reproducibility
            np.random.seed(123)

            df = pd.read_csv("../INFC_2015_climatic.csv")
            X = df[CONFIG.features]
            y = df[TARGET]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

            model_class, param_distributions, search_cv_args = MODEL
            
            if param_distributions is None:
                
                regressor = model_class(_)
            
            else:
                
                search_cv = RandomizedSearchCV(
                    model_class(_), param_distributions=param_distributions,
                    scoring="neg_mean_squared_error", random_state=0, **search_cv_args
                )
                search_cv.fit(X_train, y_train)

                print("The best hyperparameters are ",search_cv.best_params_)
                
                if type(search_cv.estimator) not in [
                    KNeighborsRegressor,
                    SVR
                ]:
                    regressor = model_class(_).set_params( # use search_cv.estimator, to make it independent from the estimator's class
                        random_state=0,           # fixed random state
                        **search_cv.best_params_, # pass all parameters without to need to manually assign them
                    )
                else:
                    regressor = model_class(_).set_params( # use search_cv.estimator, to make it independent from the estimator's class
                        **search_cv.best_params_, # pass all parameters without to need to manually assign them
                    )

            data = []
            
            # =================================================================
            # the following code is an alternative, to split the dataset in
            # 66/33 random train/test splits for 10 times to run the evaluation
            
            folds = []

            for random_state in range(10):
                cv = KFold(n_splits=4, random_state=random_state, shuffle=True)
                tmp_folds = cv.split(X)
                folds.append(next(tmp_folds))
                

            for i, (train_index, test_index) in enumerate(folds):

                X_train = X.iloc[train_index]
                y_train = y.iloc[train_index].values

                X_test = X.iloc[test_index]
                y_test = y.iloc[test_index].values

                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)

                data.append({
                    "target": TARGET,
                    "config": CONFIG.name,
                    "model": regressor.__class__.__name__,
                    "hyperparams": None if param_distributions is None else search_cv.best_params_,
                    "fold": i,
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_test": X_test,
                    "y_test": y_test,
                    "y_pred": y_pred,
                    "model_obj": deepcopy(regressor),
                })

            data = pd.DataFrame(data)
            save_path = f"results/predictions--{TARGET}--{CONFIG.name}--{regressor.__class__.__name__}.pickle"
            data.to_pickle(save_path)
            # print("predictions saved to", save_path)
            # display(data)

            data["MSE"] = data.apply(lambda row: mean_squared_error(row.y_test, row.y_pred), axis=1)
            data["R2"] = data.apply(lambda row: r2_score(row.y_test, row.y_pred), axis=1)
            data["MAPE"] = data.apply(lambda row: mean_absolute_percentage_error(row.y_test, row.y_pred), axis=1)
            data["RMSE"] = data.apply(lambda row: root_mean_squared_error(row.y_test, row.y_pred), axis=1)
            data["NRMSE"] = data.apply(lambda row: normalized_root_mean_squared_error(row.y_test, row.y_pred, norm_factor=row.y_train.mean()), axis=1)

            data = data.drop(columns=["X_train", "y_train", "X_test", "y_test", "y_pred"])
            save_path = f"results/metrics--{TARGET}--{CONFIG.name}--{regressor.__class__.__name__}.pickle"
            data.to_pickle(save_path)
            print("metrics, predictions and models saved to\n", save_path)
            # display(data)

            for metric in ["R2", "MAPE", "RMSE", "NRMSE"]:

                print(f"{metric:>10} {data[metric].mean().round(2):>7} ± {data[metric].std().round(2):>5}")


# In[2]:


import shutil

if os.path.exists("catboost_info"):
    shutil.rmtree("catboost_info")


# In[ ]:




