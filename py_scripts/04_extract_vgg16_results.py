#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from glob import glob


def get_metadata(path):
    
    file_name = path.split("/")[-1][:-4]
    target, config, model, weights, freeze, batch_size, lr, epochs, fold = file_name.split("--")
    item = dict(
        target=target,
        config=config,
        model=model,
        weights=weights,
        freeze=freeze,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        fold=fold,
        shortcode="--".join(file_name.split("--")[:-1]),
    )
    return item


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


# In[2]:


for TARGET in ["CS", "CSE"]:

    paths = glob(f"dl_output/{TARGET}--*--vgg16--DEFAULT--False--8--0.001--50*.pkl")

    dfs = []
    for p in paths:
        df = pd.read_pickle(p)

        data = get_metadata(p)
        for k,v in data.items():
            df[k] = v

        df = df[~df.config.str.contains("Conf5")]

        dfs.append(df)

    df = pd.concat(dfs)


    MODEL = "vgg16"
    tmp_idx = df.shortcode.str.split("--").str[2:].str.join("--") == MODEL+"--DEFAULT--False--8--0.001--50"
    gb = df[tmp_idx].groupby(["config", "fold"])
    best = []
    for (config, fold), g in gb:
        idxmax = g.test_R2.astype(float).idxmax()
        best.append(g.loc[idxmax])
    best = pd.concat(best, axis=1, ignore_index=True).T
    best_mean = best.groupby("shortcode").agg({
        "test_R2": "mean",
        "train_R2": "mean",
        "test_RMSE": "mean",
        "train_RMSE": "mean",
        "test_%RMSE": "mean",
        "train_%RMSE": "mean",

    })
    best_std = best.groupby("shortcode").agg({
        "test_R2": "std",
        "train_R2": "std",

        "test_RMSE": "std",
        "train_RMSE": "std",
        "test_%RMSE": "std",
        "train_%RMSE": "std",

    })
    for col in best_mean.columns:
        best_mean[col] = best_mean[col].astype(str).str[:5] + " +/- " + best_std[col].astype(str).str[:5]

    print("="*40)
    print(MODEL.upper(), "|", "DEFAULT--False--8--0.001--50")
    best_mean.index=best_mean.index.str.split("--").str[1]
    display(best_mean)

    data = best
    data["MSE"] = data.apply(lambda row: mean_squared_error(row.test_real, row.test_pred), axis=1)
    data["R2"] = data.apply(lambda row: r2_score(row.test_real, row.test_pred), axis=1)
    data["MAPE"] = data.apply(lambda row: mean_absolute_percentage_error(row.test_real, row.test_pred), axis=1)
    data["RMSE"] = data.apply(lambda row: root_mean_squared_error(row.test_real, row.test_pred), axis=1)
    data["NRMSE"] = data.apply(lambda row: normalized_root_mean_squared_error(row.test_real, row.test_pred, norm_factor=row.train_real.mean()), axis=1)

    data["MSE_train"] = data.apply(lambda row: mean_squared_error(row.train_real, row.train_pred), axis=1)
    data["R2_train"] = data.apply(lambda row: r2_score(row.train_real, row.train_pred), axis=1)
    data["MAPE_train"] = data.apply(lambda row: mean_absolute_percentage_error(row.train_real, row.train_pred), axis=1)
    data["RMSE_train"] = data.apply(lambda row: root_mean_squared_error(row.train_real, row.train_pred), axis=1)
    data["NRMSE_train"] = data.apply(lambda row: normalized_root_mean_squared_error(row.train_real, row.train_pred, norm_factor=row.train_real.mean()), axis=1)
    data

    for _, d in data.groupby(["target", "config"]):
        d["hyperparams"] = None
        d["model_obj"] = None
        d["model"] = "DeepCNN"
        d["model_name"] = None
        d = d[['target',
         'config',
         'model_name',
         'model',
         'hyperparams',
         'fold',
         'model_obj',
         'MSE',
         'R2',
         'MAPE',
         'RMSE',
         'NRMSE',
         'MSE_train',
         'R2_train',
         'MAPE_train',
         'RMSE_train',
         'NRMSE_train']]
        display(d)
        save_path = f"../results/metrics--{d.target.iloc[0]}--{d.config.iloc[0]}--DeepCNN.pickle"
        d.to_pickle(save_path)
        print("metrics, predictions and models saved to\n", save_path)


# In[ ]:





# In[ ]:




