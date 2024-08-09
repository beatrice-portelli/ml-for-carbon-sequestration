#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from configs import *
from torchvision import models
from dl_experiment import launch_experiment

# fixed

PATCH_DIM = 32
DEVICE = "cuda"
EPOCHS = 50

# varying

TARGETs = [
    TARGET_CARBON_STORAGE,
    TARGET_CARBON_SEQUESTRATION,
]

CONFIGs = [
    DL_CONFIG_sat,
    DL_CONFIG_sat_clim,
    DL_CONFIG_sat_chm,
    DL_CONFIG_sat_clim_chm,
]
BATCH_SIZEs = [16,8,4]
LRs = [1e-2, 1e-3, 1e-4]
FREEZEs = [True, False]
MODEL_CLASSs = [
    models.vgg16,
    # models.vgg11,
    # models.resnet18,
    # models.resnet50,
    # models.efficientnet_v2_s,
    # models.mobilenet_v3_small,
]
PRETRAINED_WEIGHTSs = ["DEFAULT", None]

total_experiments = len(CONFIGs)*len(BATCH_SIZEs)*len(LRs)*len(FREEZEs)*len(MODEL_CLASSs)*len(PRETRAINED_WEIGHTSs)*len(TARGETs)
cnt = 0

for TARGET in TARGETs:
    for BATCH_SIZE in BATCH_SIZEs:
        for LR in LRs:
            for MODEL_CLASS in MODEL_CLASSs:
                for FREEZE in FREEZEs:
                    for PRETRAINED_WEIGHTS in PRETRAINED_WEIGHTSs:
                        for CONFIG in CONFIGs:
                            cnt+=1
                            print("="*20)
                            print(cnt, "/", total_experiments)
                            launch_experiment(
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
                            )

