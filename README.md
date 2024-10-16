# Accompanying software and data for the paper "Assessing Ensemble Models for Carbon Sequestration and Storage Estimation in Forests Using Remote Sensing Data"

Authors:
Mehdi Fasihi ¹,
Beatrice Portelli ¹ ²,
Luca Cadez ³ ⁴,
Antonio Tomao ³,
Alex Falcon ¹,
Giorgio Alberti ³,
Giuseppe Serra ¹.

¹ Department of Mathematics, Computer Science and Physics, University of Udine, Udine, Italy <br>
² Department of Biology, University of Napoli Federico II, Napoli, Italy <br>
³ Department of Agricultural, Food, Environmental and Animal Sciences, University of Udine, Udine, Italy <br>
⁴ Department of Life Sciences, University of Trieste, Trieste, Italy

Paper published on "Ecological Informatics": https://doi.org/10.1016/j.ecoinf.2024.102828

```
@article{FASIHI-2024-Assessing,
  title = {Assessing ensemble models for carbon sequestration and storage estimation in forests using remote sensing data},
  journal = {Ecological Informatics},
  volume = {83},
  pages = {102828},
  year = {2024},
  issn = {1574-9541},
  doi = {https://doi.org/10.1016/j.ecoinf.2024.102828},
  url = {https://www.sciencedirect.com/science/article/pii/S1574954124003704},
  author = {Mehdi Fasihi and Beatrice Portelli and Luca Cadez and Antonio Tomao and Alex Falcon and Giorgio Alberti and Giuseppe Serra},
}
```

Repository archived on Zenodo: https://zenodo.org/doi/10.5281/zenodo.10932817

```
@misc{PORTELLI-2024-Assessing,
  doi = {10.5281/zenodo.13285468},
  author = {Beatrice Portelli and Mehdi Fasihi and Luca Cadez and Antonio Tomao and Alex Falcon and Giorgio Alberti and Giuseppe Serra},
  note = {Software published on Zenodo},
  title = {Accompanying software and data for the paper "Assessing Ensemble Models for Carbon Sequestration and Storage Estimation in Forests Using Remote Sensing Data"},
  year = {2024}
}
```

## Metadata

See file `metadata.xml`

## Requirements

All experiments were run using **python 3.8.13**.

For library requirements check `requirements.txt`


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

#### `dl_experiment.py`
Contains the main functions to train the DeepCNN model.

#### `dl_utils.py`
Contains helper functions to initialize the DeepCNN model and prepare its input data.



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

#### `01_report_hyperparams.*`
Saves the hyperparameters of all the best models in
  - `../results/hyperparams--<TARGET>.[txt/csv/xlsx]`
  - `../figures_and_tables/table_appendix_hyperparameters--<TARGET>.[txt/csv/xlsx]`

#### `02_calculate_ensemble.*`
Calculates the predictions of the Ensemble model consisting of the following base models: RF, CatBoost, LightGBM, AdaBoost. Saves predictions and metrics in
  - `../results/predictions--<TARGET>--<CONFIG>--Ensemble.pickle`
  - `../results/predictions--<TARGET>--<CONFIG>--Ensemble.csv`
  - `../results/metrics--<TARGET>--<CONFIG>--Ensemble.pickle`


#### `03_grid_search_DeepCNN.*`
Contains the code to run the training and evaluation of the DeepCNN models using the configurations contained in `experiments_to_run.py`.
The results of the training are saved in the folder `dl_output`

#### `04_extract_vgg16_results.*`
Extracts the results of the best DeepCNN model (VGG16) and converts them in the same format used by the machine learning models.
Saves the metrics in
  - `../results/metrics--<TARGET>--<CONFIG>--DeepCNN.pickle`

#### `05_create_results_table.*`
Loads all metrics file available in the `../results` folder,
computes the average metrics (and std).
Saves the resulting table(s) in
  - `../results/table_<TARGET>.[txt/csv/xlsx]`
  - `../figures_and_tables/table_performance_<TARGET>.[txt/csv/xlsx]`

#### `06_plot_trends.*`
Loads all the predictions, plots the correlation between real and predicted values and their marginal distributions.
Saves the results for the Ensemble model on Conf3 in
  - `../results/trends_with_margins_▸ Ensemble_CS_Conf3.[png/pdf]`
  - `../figures_and_tables/figure_comparison_real_predicted_data_▸ Ensemble_CS_Conf3.png`

#### `07_plot_comparison.*`
Loads all the metrics and creates the figures to showcase the performance diversity of all the models.
Figures are saved in
  - `../figures_and_tables/figure_performance_comparison_<TARGET>_<CONFIG>.png`

#### `08_shap.*`
Computes the SHAP feature importances for the Random Forest (RF) on all targets and configurations.
Saves different kinds of plots as
  - `../figures_and_tables/figure_shap_BAR_RF_<TARGET>_<CONFIG>.png`
  - `../figures_and_tables/figure_shap_FORCEbad_RF_<TARGET>_<CONFIG>.png`
  - `../figures_and_tables/figure_shap_FORCEgood_RF_<TARGET>_<CONFIG>.png`
  - `../figures_and_tables/figure_shap_SUMMARY_RF_<TARGET>_<CONFIG>.png`
  - `../figures_and_tables/figure_shap_VIOLIN_RF_<TARGET>_<CONFIG>.png`



## To replicate the experiments

1) Download all files
2) Verify that your python version is 3.8.13
3) Install the required libraries using `pip install -r requirements.txt`
4) Move to the `py_scripts` or `ipynb_notebooks` folder
5) Execute all files in sequence (00, 01, 02, 03, 04, ...)
6) The figures and tables present in the manuscript can be found in the folder `figures_and_tables`.<br>
Detailed results can be found in the `results` folder.

**Note 1**: despite fixing all library versions and random seeds, some models will produce differents results every time the code is run.
This is a known issue in the scikit-learn library and the authors could not improve their reproducibility any further.
The models affected by this issue are: RandomForestRegressor (RF), GradientBoostingRegressor (GBDT), and CatBoostRegressor (CatBoost).
However, the performance is always close to the mean and standard deviation reported in the original experiments and does not impact our findings.

**Note 2**: by default, script `00_run_all_experiments.[py/ipynb]` will **not** overwrite the results already present in the `result` folder and will **not** train new models if they are already present. If you want to train new models you can:
- rename, move, or delete the `results` folder before running the scripts
- or, replace `OVERWRITE = False` with `OVERWRITE = True` in `00_run_all_experiments.[py/ipynb]`



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

### `spatial_clusters_10.csv`

For each sample of the dataset (279 rows), code of the spatial cluster it belongs to.
Samples are in the same order as `data.csv`.
There are 10 clusters numbered from 0 to 9.

### `preprocess_images_to_samples.ipynb`

Code used to generate the input patches for the 279 samples of the dataset.
The outputs are saved in the `image_data` folder.

### `image_data/32x32.[csv/pkl]`

Table containing the 32x32 matrices of image-like input data for the DeepCNN models, for the 12 input features.

Table dimensions:
- 8928 rows (279 samples * 32 image size) + 1 header (feature names)
- 384 columns (12 input features * 32 image size) + 1 index (sample id, same as `data.csv`)

Preview of the data:

```
index    NDVI      NDVI  ...         ASP         ASP
0    0.803525  0.793218  ...  151.813477  163.113388
0    0.818427  0.811217  ...  159.014526  167.207336
0    0.824016  0.804485  ...  160.495529  169.547455
0    0.819101  0.806417  ...  161.815430  173.306885
0    0.807702  0.811035  ...  162.627441  174.042145
..        ...       ...  ...         ...         ...
278  0.845726  0.830755  ...  256.716888  244.695602
278  0.840384  0.836642  ...  249.973343  231.477371
278  0.834502  0.836623  ...  255.554337  221.256027
278  0.834502  0.836623  ...  266.176819  217.352722
278  0.835701  0.821787  ...  270.000000  316.146881
```

Schema of the data:

```
        column# → 0 ...   31   32 ...   63 ... 352  ... 383 
       ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
row# ↴ ┃ index NDVI ... NDVI NDII ... NDII ... ASP  ... ASP 
     0 ┃   0   ┌────---────┐ ┌────---────┐     ┌────---────┐
     1 ┃   0   │           │ │           │     │           │
     2 ┃   0   │ NDVI      │ │ NDII      │     │ ASP       │
     : ┃   :   : channel   : : channel   : ... : channel   :
    29 ┃   0   │ sample 0  │ │ sample 0  │     │ sample 0  │
    30 ┃   0   │           │ │           │     │           │
    31 ┃   0   └────---────┘ └────---────┘     └────---────┘
    32 ┃   1   ┌────---────┐ ┌────---────┐     ┌────---────┐
    33 ┃   1   │           │ │           │     │           │
    34 ┃   1   │ NDVI      │ │ NDII      │     │ ASP       │
     : ┃   :   : channel   : : channel   : ... : channel   :
    61 ┃   1   │ sample 1  │ │ sample 1  │     │ sample 1  │
    62 ┃   1   │           │ │           │     │           │
    63 ┃   1   └────---────┘ └────---────┘     └────---────┘
     : ┃   :         :             :                 :
  8896 ┃  278  ┌────---────┐ ┌────---────┐     ┌────---────┐
  8897 ┃  278  │           │ │           │     │           │
  8898 ┃  278  │ NDVI      │ │ NDII      │     │ ASP       │
     : ┃   :   : channel   : : channel   : ... : channel   :
  8925 ┃  278  │ samp. 278 │ │ sam. 278  │     │ samp. 278 │
  8926 ┃  278  │           │ │           │     │           │
  8927 ┃  278  └────---────┘ └────---────┘     └────---────┘
       ┃
```


