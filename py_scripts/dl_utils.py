import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torchvision import models


import os 
import random
import numpy as np 

DEFAULT_RANDOM_SEED = 2021

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
# tensorflow random seed 
# import tensorflow as tf 
# def seedTF(seed=DEFAULT_RANDOM_SEED):
#     tf.random.set_seed(seed)
    
# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
# basic + tensorflow + torch 
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    # seedTF(seed)
    seedTorch(seed)



def get_folds():

    random.seed(123)

    # ============== spatial clustering, 10 clusters, TRAINING [start]
    cluster_df = pd.read_csv("../spatial_clusters_10.csv")

    tmp_test_folds = []
    test_folds = []
    train_folds = []

    for cluster_idx, data in cluster_df.groupby("cluster"):
        tmp_test_folds.append(data.index.tolist())

    random.shuffle(tmp_test_folds)
    for i in range(5):
        test_folds.append(tmp_test_folds[i*2]+tmp_test_folds[i*2+1])
        train_folds.append(
            list(set(cluster_df.index.tolist())-set(test_folds[-1]))
        )

    [random.shuffle(x) for x in test_folds]
    [random.shuffle(x) for x in train_folds]
    # ============== spatial clustering, 10 clusters, TRAINING [end]

    # ============== spatial clustering, 10 clusters, VALIDATION [start]
    tuning_test_folds = []
    tuning_train_folds = []

    random.shuffle(tmp_test_folds)
    for i in range(5):
        tuning_test_folds.append(tmp_test_folds[i*2]+tmp_test_folds[i*2+1])
        tuning_train_folds.append(
            list(set(cluster_df.index.tolist())-set(tuning_test_folds[-1]))
        )

    [random.shuffle(x) for x in tuning_test_folds]
    [random.shuffle(x) for x in tuning_train_folds]
    # ============== spatial clustering, 10 clusters, VALIDATION [end]
    
    return train_folds, test_folds, tuning_train_folds, tuning_test_folds





def get_dataloaders_and_scalers(X, y, train_index, test_index, CONFIG, BATCH_SIZE, PATCH_DIM):
    
    X_train = X.loc[train_index]
    y_train = y.iloc[train_index].values

    X_test = X.loc[test_index]
    y_test = y.iloc[test_index].values
    
    scalers = {
        feature: {
            "mean": X_train[feature].values.flatten().mean(),
            "std": X_train[feature].values.flatten().std(),
        }
        for feature in CONFIG.features
    }
    scaler_y = {
        "mean": y_train.flatten().mean(),
        "std": y_train.flatten().std(),
    }
    
    # scale X
    X_train = X_train.apply(
        lambda col: (col-scalers[col.name]["mean"])/scalers[col.name]["std"]
    )
    X_test = X_test.apply(
        lambda col: (col-scalers[col.name]["mean"])/scalers[col.name]["std"]
    )
        
    # scale y
    y_train = (y_train - scaler_y["mean"])/scaler_y["std"]
    y_test = (y_test - scaler_y["mean"])/scaler_y["std"]
        
    
    X_train = X_train.values.reshape(
        len(y_train), # num samples
        PATCH_DIM,    # img dim
        PATCH_DIM,    # img dim
        -1, # num bands / features
    )
    y_train = y_train.reshape(len(y_train), 1)
    
    X_test = X_test.values.reshape(
        len(y_test), # num samples
        PATCH_DIM,    # img dim
        PATCH_DIM,    # img dim
        -1, # num bands / features
    )
    y_test = y_test.reshape(len(y_test), 1)
    
    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train).transpose(2,3).transpose(1,2),
        torch.FloatTensor(y_train),
    )
    
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test).transpose(2,3).transpose(1,2),
        torch.FloatTensor(y_test)
    )
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    return trainloader, testloader, scalers, scaler_y



def init_model(MODEL_CLASS, PRETRAINED_WEIGHTS, CONFIG, FREEZE, DEVICE):
    
    model = MODEL_CLASS(weights=PRETRAINED_WEIGHTS)
    if FREEZE:
        cnt=0
        for param in model.parameters():
            if param.requires_grad:
                    cnt+=1
            param.requires_grad = False

    if len(CONFIG.features)!=3:

        if isinstance(model, models.VGG):

            first_conv = model.features[0]

            model.features[0] = torch.nn.Conv2d(
                in_channels=len(CONFIG.features),
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False if type(first_conv.bias)==bool and not first_conv.bias else True
            )

        elif isinstance(model, models.ResNet):

            first_conv = model.conv1

            model.conv1 = nn.Conv2d(
                in_channels=len(CONFIG.features),
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False if type(first_conv.bias)==bool and not first_conv.bias else True
            )
            
        elif isinstance(model, models.MobileNetV3) or isinstance(model, models.EfficientNet):
            
            first_conv = model.features[0][0]
            
            model.features[0][0] = torch.nn.Conv2d(
                in_channels=len(CONFIG.features),
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False if type(first_conv.bias)==bool and not first_conv.bias else True
            )
            
        else:
            assert False, "No rules to adapt this model's first convolution to multiple channels"
            

    

    if isinstance(model, models.VGG) or isinstance(model, models.MobileNetV3) or isinstance(model, models.EfficientNet):
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 1, bias=True)
    elif isinstance(model, models.ResNet):
        model.fc = torch.nn.Linear(model.fc.in_features, 1, bias=True)
    else:
        assert False, "No rules to adapt this model's classifier to 1 output"

    cnt=0
    for param in model.parameters():
         if param.requires_grad:
                cnt+=1

    model.to(DEVICE)
    return model


def train_one_epoch(model, trainloader, DEVICE, criterion, optimizer, scaler_y=None):
    
    model.train()
    running_loss = 0.0
    real = []
    pred = []
    
    for i, data in enumerate(trainloader):
        
        inputs, labels = data
        inputs=inputs.to(DEVICE)
        labels=labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        real += labels.detach().flatten().tolist()
        pred += outputs.detach().flatten().tolist()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    real = np.array(real)
    pred = np.array(pred)
    
    if scaler_y:
        real = unscale_y(real, scaler_y)
        pred = unscale_y(pred, scaler_y)
    
    loss = running_loss / len(trainloader)
    
    return loss, real, pred


def eval_on(model, testloader, DEVICE, criterion=None, scaler_y=None):
    
    model.eval()
    running_loss = 0.0
    real = []
    pred = []
    
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs=inputs.to(DEVICE)
        labels=labels.to(DEVICE)
        with torch.no_grad():
            outputs = model(inputs)
        if criterion:
            loss = criterion(outputs, labels)
            running_loss += loss.item()
        else:
            running_loss = -1
        real += labels.flatten().tolist()
        pred += outputs.flatten().tolist()
        
    real = np.array(real)
    pred = np.array(pred)
    
    if scaler_y:
        real = unscale_y(real, scaler_y)
        pred = unscale_y(pred, scaler_y)
        
    loss = running_loss / len(testloader) if running_loss!=-1 else -1
        
    return loss, real, pred


def unscale_y(data, scaler_y):
    return scaler_y["mean"] + (data * scaler_y["std"])