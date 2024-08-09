import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

import os

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


import pandas as pd
import random

from dl_utils import get_folds, get_dataloaders_and_scalers, init_model, train_one_epoch, seedEverything, eval_on


def launch_experiment(
    PATCH_DIM,
    TARGET,
    CONFIG,
    BATCH_SIZE,
    EPOCHS,
    LR,
    DEVICE,
    FREEZE,
    MODEL_CLASS,
    PRETRAINED_WEIGHTS,
):

    train_folds, test_folds, _, _ = get_folds()

    path = f"../image_data/{PATCH_DIM}x{PATCH_DIM}.pkl"
    df = pd.read_pickle(path)
    labels = pd.read_csv("../data.csv")[TARGET]
    df = df[CONFIG.features]
    X = df
    y = labels
    folds = zip(train_folds, test_folds)
    seedEverything(42)

    for FOLD_NUM, (train_index, test_index) in enumerate(folds):
        
        file_path = f"{TARGET}--{CONFIG.name}--{MODEL_CLASS.__name__}--{PRETRAINED_WEIGHTS}--{FREEZE}--{BATCH_SIZE}--{LR}--{EPOCHS}--{FOLD_NUM}"
        print(file_path)
        if os.path.exists(f"dl_output/{file_path}.pkl"):
            print("Skipping, file already exists.")
            continue

        trainloader, testloader, scalers, scaler_y = \
        get_dataloaders_and_scalers(X, y, train_index, test_index, CONFIG, BATCH_SIZE, PATCH_DIM)

        model = init_model(MODEL_CLASS, PRETRAINED_WEIGHTS, CONFIG, FREEZE, DEVICE)


        criterion = nn.MSELoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=LR,
            momentum=0.9,
        )

        df = []

        for epoch in tqdm(range(EPOCHS)):

            train_loss, train_real, train_pred = train_one_epoch(model, trainloader, DEVICE, criterion, optimizer, scaler_y)
            test_loss, test_real, test_pred = eval_on(model, testloader, DEVICE, criterion, scaler_y)

            item = {
                "epoch": epoch+1,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_real": train_real,
                "train_pred": train_pred,
                "test_real": test_real,
                "test_pred": test_pred,
            }
            df.append(item)    

        df = pd.DataFrame(df)
        
        try:
            df["train_R2"] = df.apply(lambda row: r2_score(row.train_real, row.train_pred) if (row.train_loss != np.inf) and (not np.isnan(row.train_loss)) else None, axis=1)
        except:
            df["train_R2"] = None
        try:
            df["test_R2"] = df.apply(lambda row: r2_score(row.test_real, row.test_pred) if (row.test_loss != np.inf) and (not np.isnan(row.test_loss)) else None, axis=1)
        except:
            df["test_R2"] = None

        try:
            df["train_RMSE"] = df.apply(lambda row: root_mean_squared_error(row.train_real, row.train_pred) if (row.train_loss != np.inf) and (not np.isnan(row.train_loss)) else None, axis=1)
        except:
            df["train_RMSE"] = None
        try:
            df["test_RMSE"] = df.apply(lambda row: root_mean_squared_error(row.test_real, row.test_pred) if (row.test_loss != np.inf) and (not np.isnan(row.test_loss)) else None, axis=1)
        except:
            df["test_RMSE"] = None

        try:
            df["train_%RMSE"] = df.apply(lambda row: normalized_root_mean_squared_error(row.train_real, row.train_pred, row.train_real.mean()) if (row.train_loss != np.inf) and (not np.isnan(row.train_loss)) else None, axis=1)
        except:
            df["train_%RMSE"] = None
        try:
            df["test_%RMSE"] = df.apply(lambda row: normalized_root_mean_squared_error(row.test_real, row.test_pred, row.train_real.mean()) if (row.test_loss != np.inf) and (not np.isnan(row.test_loss)) else None, axis=1)
        except:
            df["test_%RMSE"] = None


        df.drop(columns=["train_real", "train_pred", "test_real", "test_pred"]).to_csv(f"dl_output/{file_path}.csv", index=False)
        df.to_pickle(f"dl_output/{file_path}.pkl")
                