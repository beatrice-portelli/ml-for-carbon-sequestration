# ML for Carbon Sequestration

Data and code for the manuscript
"Assessing Ensemble Models for Carbon Sequestration and Storage Estimation in Forests Using Remote Sensing Data".

Authors:
Mehdi Fasihi ¹,
Beatrice Portelli ¹ ²,
Luca Cadez ³ ⁴,
Antonio Tomao ³,
Giorgio Alberti ³ ⁵,
Giuseppe Serra ¹.

¹ Department of Mathematics, Computer Science and Physics, University of Udine, Udine, Italy <br>
² Department of Biology, University of Napoli Federico II, Napoli, Italy <br>
³ Department of Agricultural, Food, Environmental and Animal Sciences, University of Udine, Udine, Italy <br>
⁴ Department of Life Sciences, University of Trieste, Trieste, Italy <br>
⁵ Free University of Bolzano, Bolzano, Italy

## Metadata

See file `metadata.xml`

## Requirements

All experiments were run using **python 3.8.13**



## Data overview

### `data.csv`

Data used for the experiments, 279 rows (samples), 42 columns.

Column descriptions:

```
* CSE
  Carbon Sequestration (target value)
  measured in tC ha-1 yr-1
  
* CS
  Carbon Storage (target value)
  measured in tC ha-1
  
* NDVI_[max/mean/median/stdDev]
  Values for the NDVI calculated on the satellite images of Sentinel 2
  (max/mean/median/standard deviation)
  
* NDII_[max/mean/median/stdDev]
  Values for the NDII calculated on the satellite images of Sentinel 2
  (max/mean/median/standard deviation)
  
* GNDVI_[max/mean/median/stdDev]
  Values for the GNDVI calculated on the satellite images of Sentinel 2
  (max/mean/median/standard deviation)
  
* EVI_[max/mean/median/stdDev]
  Values for the EVI calculated on the satellite images of Sentinel 2
  (max/mean/median/standard deviation)
  
* chm_[max/mean/median/stdev]
  Values for the Canopy Height Model (DSM first-DTM)
  (max/mean/median/standard deviation)
  
* elevation_[max/mean/median/stdev]
  Digital Terrain Model at 10 m
  (max/mean/median/standard deviation)
  
* slope_percentage_[max/mean/median/stdev]
  Values for the slope (percentage)
  (max/mean/median/standard deviation)
  
* aspect_degree_[max/mean/median/stdev]
  Values for the aspect (degree)
  (max/mean/median/standard deviation)
  
* pr_avg_JJA_median
  Median precipitation during summer months (June, July, August)
  (mean)
  
* pr_avg_MAM_median
  Median precipitation during spring months (March, April, May)
  (mean)
  
* tas_[avg/max/min]_JJA_median
  Median precipitation during summer months (June, July, August)
  (mean, max, min)
  
* tas_[avg/max/min]_MAM_median
  Median temperature at soil during spring months (March, April, May)
  (mean, max, min)
  
```



## Code overview

### `results`

Contains the outputs of the models described in the paper.
  
### `figures_and_tables`

Contains the figures and tables contained in the manuscript that can be generated using the code in this repository.

### `py_scripts` and `ipynb_notebooks`

Contain the .py (python script) version and .ipynb (python notebook) version of all the code needed to replicate the experiments.

The following are the details on the code files:

#### `configs.py`
Contains constants such as:
- names of the target variables (denoted as `TARGET_*`)
- definitions of the input configurations (denoted as `CONFIG_*`)

#### `experiments_to_run.py`
Defines:
- `MODELS`: the full list of models to test and their hyperparameters
- `CONFIGS`: the configudations of input features to test (Conf1, Conf2, Conf3, and Conf4 in the manuscript)
- `TARGETS`: list of all the targets to predict (CS and CSE)

#### `00_run_all_experiments.*`
Contains the code to run the training and evaluation of the models and configurations contained in `experiments_to_run.py`. The code does the following for each model:
- load `data.csv` and select the relevant subset of input features (according to the CONFIG) and target (according to TARGET)
- perform a randomized grid search to find the best hyperparameters for the model
- train and evaluate the model with 5-fold cross validation
- calculate the following metrics on all the folds: MSE, R2, MAPE, RMSE, NRMSE (%RMSE)
- save the predictions and metrics in
  - `../results/predictions--<TARGET>--<CONFIG>--<MODEL>.pickle`
  - `../results/predictions--<TARGET>--<CONFIG>--<MODEL>.csv`
  - `../results/metrics--<TARGET>--<CONFIG>--<MODEL>.pickle`

#### `01_summarize_all_metrics.*`
Saves the hyperparameters of all the best models in
  - `../results/hyperparams--<TARGET>.[txt/csv/xlsx]`
  - `../figures_and_tables/table_appendix_hyperparameters--<TARGET>.[txt/csv/xlsx]`

#### `02_create_results_table.*`
Loads all metrics file available in the `../results` folder,
computes the average metrics (and std).
Saves the resulting table(s) in
  - `../results/table_<TARGET>.[txt/csv/xlsx]`
  - `../figures_and_tables/table_performance_<TARGET>.[txt/csv/xlsx]`

#### ⚠️ `03_plot_comparison.*` new, work in progress ⚠️


#### ⚠️ `03_feature_importance.*` removed, work in progress ⚠️
Computes the feature importances and creates the feature importance plots.
Saves all features importances in
  - `../results/feature_importances.csv`

Saves the images for the Random Forest (RF) on Conf3 in
  - `../results/feature_importance_RF_<TARGET>.[pdf/png]`
  - `../results/feature_importance_RF_<TARGET>_rot90.png`
  - `../figures_and_tables/figure_feature_importance_RF_<TARGET>.png`

#### ⚠️ `04_plot_trends.*` removed, work in progress ⚠️
Loads all the predictions, plots the correlation between real and predicted values and their marginal distributions.
Saves the results for the Random Forest (RF) on Conf3 in
  - `../results/trends_with_margins_RF_<TARGET>.[pdf/png]`
  - `../figures_and_tables/figure_comparison_real_predicted_data_RF_<TARGET>.[pdf/png]`



## To replicate the experiments

1) Download all files
2) Verify that your python version is 3.8.13
3) Install the required libraries using `pip install -r requirements.txt`
4) Move to the `py_scripts` or `ipynb_notebooks` folder
5) Execute all files in sequence (00, 01, 02, 03, 04)
6) The figures and tables present in the manuscript can be found in the folder `figures_and_tables`.<br>
Detailed results can be found in the `results` folder.

**Note 1**: despite fixing all library versions and random seeds, some models will produce differents results every time the code is run.
This is a known issue in the scikit-learn library and the authors could not improve their reproducibility any further.
The models affected by this issue are: RandomForestRegressor (RF), GradientBoostingRegressor (GBDT), and CatBoostRegressor (CatBoost).
However, the performance is always close to the mean and standard deviation reported in the original experiments and does not impact our findings.

**Note 2**: by default, script `00_run_all_experiments.[py/ipynb]` will **not** overwrite the results already present in the `result` folder and will **not** train new models if they are already present. If you want to train new models you can:
- rename, move, or delete the `results` folder before running the scripts
- or, replace `OVERWRITE = False` with `OVERWRITE = True` in `00_run_all_experiments.[py/ipynb]`

