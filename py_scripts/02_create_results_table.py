#!/usr/bin/env python
# coding: utf-8

# In[1]:


root_folder = "results"


# In[2]:


model2paper = {
    'CatBoostRegressor':         "CatBoost",
    'GradientBoostingRegressor': "GBDT",
    'KNeighborsRegressor':       "KNN",
    'LinearRegression':          "MLR",
    'MLPRegressor':              "MLP",
    'RandomForestRegressor':     "RF",
    'SVR':                       "SVR",
    'XGBRegressor':              "XGBoost",
}

config2paper = {
    "CONFIG_0": "Conf1", # satellite
    "CONFIG_1": "Conf4", # satellite + climate + chm
    "CONFIG_2": "Conf3", # satellite + chm
    "CONFIG_3": "Conf2", # satellite + climate
}

metrics2paper = {
    "R2": "R2",
    "RMSE": "RMSE",
    "NRMSE": "%RMSE",
}

target2paper = {
    "ICCapv_ha": "CSE",
    "Catot_ha": "CS",
}

paper2model = {v:k for k,v in model2paper.items()}
paper2config = {v:k for k,v in config2paper.items()}
paper2metrics = {v:k for k,v in metrics2paper.items()}

model_order = [
    "CatBoost",
    "GBDT",
    "KNN",
    "MLR",
    "MLP",
    "RF",
    "SVR",
    "XGBoost",
]

config_order = [
    "Conf1",
    "Conf2",
    "Conf3",
    "Conf4",
]

metrics_order = [
    "R2",
    "RMSE",
    "%RMSE",
]


# In[3]:


import pandas as pd
from glob import glob
from configs import *

paths = glob(f'{root_folder.replace("[","[[]").replace("]","[]]").replace("[[[]]", "[[]")}/metrics*')

df = []
for path in paths:
    tmp_df = pd.read_pickle(path)
    df.append(tmp_df)
df = pd.concat(df)

metrics = ["R2", "RMSE", "NRMSE"]
mean_df = df.groupby(["target", "config", "model"]).agg("mean")[metrics]#.reset_index()
std_df = df.groupby(["target", "config", "model"]).agg("std")[metrics]#.reset_index()

std_df.columns = ["s_"+x for x in std_df.columns]
overall_df = pd.concat((mean_df, std_df), axis=1)

for metric in metrics:
    overall_df[f"str_{metric}"] = overall_df.apply(lambda row: f"{row[metric]:.2f} ± {row['s_'+metric]:.2f}", axis=1)
    
overall_df = overall_df.drop(columns = [x for x in overall_df.columns if "str" not in x])
overall_df.columns = metrics
overall_df = overall_df.reset_index()

overall_df.target.replace(target2paper, inplace=True)
overall_df.config.replace(config2paper, inplace=True)
overall_df.sort_values(["target","config", "model"], inplace=True)
overall_df.model.replace(model2paper, inplace=True)
overall_df.rename(columns=metrics2paper, inplace=True)
overall_df


# In[4]:


for target in ["CS", "CSE"]:
    tmp = overall_df[(overall_df.target==target)].drop(columns=["target"])
    print("results exported to", f"{root_folder}/table_{target}.[md/xlsx]")
    tmp.to_markdown(f"{root_folder}/table_{target}.md", index=False, tablefmt="grid")
    tmp.to_excel(f"{root_folder}/table_{target}.xlsx", index=False)


# In[ ]:




