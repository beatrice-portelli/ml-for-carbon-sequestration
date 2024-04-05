#!/usr/bin/env python
# coding: utf-8

# In[1]:


root_folder = "../results"


# In[2]:


metrics2paper = {
    "R2": "R2",
    "RMSE": "RMSE",
    "NRMSE": "%RMSE",
}


# In[3]:


import pandas as pd
from glob import glob
from experiments_to_run import *

paths = glob(f'{root_folder}/metrics*')

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

overall_df.sort_values(["target","config", "model"], inplace=True)
overall_df.rename(columns=metrics2paper, inplace=True)
overall_df


# In[4]:


for target in TARGETS:
    tmp = overall_df[(overall_df.target==target)].drop(columns=["target"])
    print("results exported to", f"{root_folder}/table_{target}.[txt/xlsx]")
    tmp.to_markdown(f"{root_folder}/table_{target}.txt", index=False, tablefmt="fancy_grid")
    tmp.to_excel(f"{root_folder}/table_{target}.xlsx", index=False)
    
    print("results exported to", f"../figures_and_tables/table_performance_{target}.[txt/xlsx/csv]")
    tmp.to_markdown(f"../figures_and_tables/table_performance_{target}.txt", index=False, tablefmt="fancy_grid")
    tmp.to_excel(f"../figures_and_tables/table_performance_{target}.xlsx", index=False)
    tmp.to_csv(f"../figures_and_tables/table_performance_{target}.csv", index=False)


# In[ ]:




