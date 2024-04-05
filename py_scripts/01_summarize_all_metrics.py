#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from glob import glob
from experiments_to_run import *


# In[2]:


paths = glob("../results/metrics*")

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

print(mean_df)
print(std_df)


# In[4]:


for target_var in TARGETS:
    
    tmp = df[(df.target == target_var)]\
      [["target", "config", "model", "hyperparams"]]\
      .groupby(["config", "model"]).agg({
        "hyperparams":"first",
      }).reset_index()

    tmp.hyperparams = tmp.hyperparams.apply(
        lambda x: "\n".join([f"{k}: {v}" for k,v in x.items()]) if x is not None else ""
    )
    
    gb = tmp.groupby("model")
    new_tmp = []
    for m,g in gb:
        new_tmp.append(g.drop(columns=["model"]).rename(columns={"hyperparams":m}).set_index("config").T)
    
    tmp = pd.concat(new_tmp).reset_index().rename(columns={"index":"Model"})
    
    print("     "+target_var)
    print("results exported to", f"../results/hyperparams_{target_var}.[txt/xlsx]")
    tmp.to_markdown(f"../results/hyperparams_{target_var}.txt", index=False, tablefmt="fancy_grid")
    tmp.to_excel(f"../results/hyperparams_{target_var}.xlsx", index=False)
    
    print("results exported to", f"../figures_and_tables/table_appendix_hyperparameters_{target_var}.[txt/xlsx/csv]")
    tmp.to_markdown(f"../figures_and_tables/table_appendix_hyperparameters_{target_var}.txt", index=False, tablefmt="fancy_grid")
    tmp.to_excel(f"../figures_and_tables/table_appendix_hyperparameters_{target_var}.xlsx", index=False)
    tmp.to_csv(f"../figures_and_tables/table_appendix_hyperparameters_{target_var}.csv", index=False)


# In[ ]:




