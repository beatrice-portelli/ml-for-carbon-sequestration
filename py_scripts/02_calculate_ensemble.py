#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from glob import glob
from experiments_to_run import *


from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def normalized_root_mean_squared_error(y_true, y_pred, norm_factor=None):
    if norm_factor is None:
        assert False, "Set norm_factor (for example the average target value for the training set)"
    rmse = root_mean_squared_error(y_true, y_pred)
    return (rmse / norm_factor)*100


ENSEMBLE_MODELS = [
    "RF",
    "CatBoost",
    "LightGBM",
    "AdaBoost",
]

ensemble_res = []

for TARGET in TARGETS:
    for CONFIG in CONFIGS:
        
        CONFIG = CONFIG.name
        
        root = "../results"
        res = None
        
        for MODEL in ENSEMBLE_MODELS:

            data = pd.read_pickle(f"{root}/predictions--{TARGET}--{CONFIG}--{MODEL}.pickle")
            if res is None:
                res = data[["y_pred"]].copy()
            else:
                res += data[["y_pred"]]

        res = res/len(ENSEMBLE_MODELS)

        for col_name in [
            "target",
            "config",
            "fold",
            "X_train",
            "y_train",
            "X_test",
            "y_test",
        ]:
            res[col_name] = data[col_name]
        res["model"] = "▸ Ensemble"
        res["model_name"] = None
        res["hyperparams"] = None
        res["model_obj"] = None
        
        res = res[data.columns.tolist()]
        
        save_path = f"{root}/predictions--{TARGET}--{CONFIG}--Ensemble.pickle"
        
        res.to_pickle(save_path)
        
        tmp = res
        y_test = pd.concat(
            [x for (i,x) in tmp[["fold","y_test"]].explode(column="y_test").groupby("fold")],
            axis=0
        ).reset_index(drop=True)
        y_pred = pd.concat(
            [x for (i,x) in tmp[["fold","y_pred"]].explode(column="y_pred").groupby("fold")],
            axis=0
        ).reset_index(drop=True)
        test_index = pd.DataFrame(
            sum(tmp.X_test.apply(lambda d: d.index.tolist()).values.tolist(), []),
            columns=["sample_idx"]
        )
        tmp2 = pd.concat(( y_test, y_pred["y_pred"], test_index), axis=1)
        tmp2.to_csv(save_path.replace("pickle", "csv"), index=None)
        
        data = res
        
        data["MSE"] = data.apply(lambda row: mean_squared_error(row.y_test, row.y_pred), axis=1)
        data["R2"] = data.apply(lambda row: r2_score(row.y_test, row.y_pred), axis=1)
        data["MAPE"] = data.apply(lambda row: mean_absolute_percentage_error(row.y_test, row.y_pred), axis=1)
        data["RMSE"] = data.apply(lambda row: root_mean_squared_error(row.y_test, row.y_pred), axis=1)
        data["NRMSE"] = data.apply(lambda row: normalized_root_mean_squared_error(row.y_test, row.y_pred, norm_factor=row.y_train.mean()), axis=1)

        data = data.drop(columns=["X_train", "y_train", "X_test", "y_test", "y_pred"])
        save_path = f"../results/metrics--{TARGET}--{CONFIG}--Ensemble.pickle"
        data.to_pickle(save_path)
        print("metrics, predictions and models saved to\n", save_path)
        


# In[ ]:




