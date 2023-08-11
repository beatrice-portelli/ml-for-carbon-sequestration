#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from glob import glob
from configs import *


# In[2]:


paths = glob("results/metrics*")

df = []
for path in paths:
    tmp_df = pd.read_pickle(path)
    df.append(tmp_df)
df = pd.concat(df)
df


# In[3]:


metrics = ["R2", "MAPE", "RMSE", "NRMSE"]
mean_df = df.groupby(["target", "config", "model"]).agg("mean")[metrics]
std_df = df.groupby(["target", "config", "model"]).agg("std")[metrics]

display(mean_df)
display(std_df)


# In[4]:


for target_var in [TARGET_CARBON, TARGET_CARBON_IC]:
    
    tmp = df[(df.target == target_var)]\
      [["target", "config", "model", "hyperparams", "R2", "RMSE", "NRMSE"]]\
      .groupby(["config", "model"]).agg({
        "hyperparams":"first",
        "R2":"mean",
        "RMSE":"mean",
        "NRMSE":"mean",
      }).reset_index()
    
    tmp_std = df[(df.target == target_var)]\
      [["target", "config", "model", "hyperparams", "R2", "RMSE", "NRMSE"]]\
      .groupby(["config", "model"]).agg({
        "R2":"std",
        "RMSE":"std",
        "NRMSE":"std",
      }).reset_index()
    
    for idx, row in tmp.iterrows():
        
        tmp.at[idx, "R2"] = f"{round(row.R2, 2):.2f} ± {round(tmp_std.at[idx, 'R2'], 2):.2f}"
        tmp.at[idx, "RMSE"] = f"{round(row.RMSE, 2):.2f} ± {round(tmp_std.at[idx, 'RMSE'], 2):.2f}"
        tmp.at[idx, "NRMSE"] = f"{round(row.NRMSE, 2):.2f} ± {round(tmp_std.at[idx, 'NRMSE'], 2):.2f}"

    tmp.hyperparams = tmp.hyperparams.apply(
        lambda x: "\n".join([f"{k}: {v}" for k,v in x.items()]) if x is not None else ""
    )
    
    tmp.rename(columns={"NRMSE": "%RMSE"}, inplace=True)
    
    print("     "+target_var)
    print("results exported to", f"results/hyperparams_{target_var}.[md/xlsx]")
    tmp.to_markdown(f"results/hyperparams_{target_var}.md", index=False, tablefmt="grid")
    tmp.to_excel(f"results/hyperparams_{target_var}.xlsx")

