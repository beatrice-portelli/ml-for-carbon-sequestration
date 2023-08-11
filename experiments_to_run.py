# from scipy.stats import loguniform

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


from configs import *


MODELS = [
    
    (
        lambda x: MLPRegressor(
            random_state=0,
            activation="relu",
            # hidden_layer_sizes=[4],
            solver="adam",
            batch_size=10,
        ),
        {
            "max_iter": [10, 20, 50],
            "learning_rate_init": [0.1, 0.01],
            "hidden_layer_sizes": [[x] for x in [2, 4, 8, 64, 128]]
        },
        dict(),
    ),
    
    (
        lambda x: SVR(kernel = 'rbf'),
        {
            'C': [1000, 100, 10, 1, 0.1],       # loguniform(1e0, 1e3),
            'gamma': [1e-5, 1e-4, 1e-3, 1e-2],  # loguniform(1e-4, 1e-3),
        },
        dict(n_iter=30),
    ),
    
    (
        lambda x: KNeighborsRegressor(),
        {
            "n_neighbors": list(range(1,31)),
        },
        dict(n_iter=30),
    ),
    
    (
        lambda x: LinearRegression(),
        None,
        None
    ),
    
    (
        lambda x: RandomForestRegressor(),
        {
            'bootstrap': [True],
            'max_depth': [100, 110, 120, 130,140,150],
            'max_features': [4, 5, 6],
            'min_samples_leaf': [4, 5, 6, 8],
            'min_samples_split': [10, 12, 14, 16],
            'n_estimators': [200, 400, 600, 1000, 1200]
        },
        dict(cv = 3, n_jobs = -1),
    ),
    
     (
        lambda x: GradientBoostingRegressor(),
        {
            "n_estimators": [1, 2, 5, 10, 20, 50, 100, 200, 500,1000,2000],
            "max_leaf_nodes": [2, 5, 10, 20, 50, 100],
            "learning_rate": [1, 0.1, 0.01, 0.001], # loguniform(0.001,0.01, 1),
            'max_depth':[1,2,4],
            'subsample':[.5,.75,1]
        },
        dict(n_iter=20, n_jobs=-1),
    ),
    
    (
        lambda x: CatBoostRegressor(loss_function='RMSE', verbose=False),
        {
            'iterations': [100, 150, 200, 250],
            'learning_rate': [0.03, 0.1],
            'depth': [2, 4, 6, 8, 10],
            'l2_leaf_reg': [0.2, 0.5, 1, 3, 4]
        },
        dict(),
    ),
    
    (
        lambda x: XGBRegressor(tree_method='gpu_hist'),
        {
            "max_depth":    [4, 5, 10],
            "n_estimators": [500, 600, 700, 800],
            "learning_rate": [0.01, 0.015, 0.02]
        },
        dict(),
    ),
    
]

CONFIGS = [
    CONFIG_0,
    CONFIG_1,
    CONFIG_2,
    CONFIG_3
]
TARGETS = [
    TARGET_CARBON,
    TARGET_CARBON_IC,
]