from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    BaggingRegressor,
    AdaBoostRegressor,
    StackingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import (Ridge, BayesianRidge)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from keras.wrappers.scikit_learn import KerasRegressor


import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
# Assuming train_data_reshaped and train_labels are already defined and loaded

def build_conv1D_model(
    optimizer='adam',
    filters1=64,
    filters2=32,
    filters3=16,
    kernel_size1=7,
    kernel_size2=3,
    kernel_size3=2,
    dropout_rate=0.5,
    num_features=0,
):
    n_timesteps = num_features
    n_features  = 1
    model = keras.Sequential(name="model_conv1D")
    model.add(keras.layers.Input(shape=(n_timesteps, n_features)))
    model.add(keras.layers.Conv1D(filters=filters1, kernel_size=kernel_size1, activation='relu', name="Conv1D_1"))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Conv1D(filters=filters2, kernel_size=kernel_size2, activation='relu', name="Conv1D_2"))
    model.add(keras.layers.Conv1D(filters=filters3, kernel_size=kernel_size3, activation='relu', name="Conv1D_3"))
    model.add(keras.layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation='relu', name="Dense_1"))
    model.add(keras.layers.Dense(n_features, name="Dense_2"))
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    print(f"""
    ----------------------------------------
    built a conv1D model with parameters:
    optimizer={optimizer},
    filters1={filters1},
    filters2={filters2},
    filters3={filters3},
    kernel_size1={kernel_size1},
    kernel_size2={kernel_size2},
    kernel_size3={kernel_size3},
    dropout_rate={dropout_rate},
    num_features={num_features},
    """)
    
    return model





from configs import *


MODELS = [
    
    # format: (model_class, param_distributions, search_cv_args)
    
    
#     (
#         # Wrap the Keras model so it can be used by scikit-learn
#         lambda x: KerasRegressor(
#             build_fn=build_conv1D_model,
#             verbose=0
#         ),        
#         {
#             'optimizer': ['RMSprop', 'Adam'],
#             'filters1': [32, 64],
#             'filters2': [16, 32],
#             'filters3': [8, 16],
#             'kernel_size1': [5, 7],
#             'kernel_size2': [3, 5],
#             'kernel_size3': [2, 3],
#             'dropout_rate': [0.3, 0.5],
#             'epochs': [100],
#             'batch_size': [10, 20],
#         },
#         dict(),
#     ),
    
    
    (
        lambda x: BaggingRegressor(
            base_estimator=DecisionTreeRegressor(random_state=42),
            random_state=42,
        ),
        {
            'n_estimators': [50, 100, 200],  # Number of trees in the ensemble
            'base_estimator__max_depth': [4, 5, 10]  # Max depth of each tree
        },
        dict(),
    ),
    
    (
        lambda x: AdaBoostRegressor(
            base_estimator=DecisionTreeRegressor(max_depth=5,random_state=42),
            random_state=42,
        ),
        {
            'n_estimators': [50, 100, 200],  # Number of trees in the ensemble
            'learning_rate': [0.01, 0.1, 1.0]  # Learning rate of the boosting process
        },
        dict(),
    ),
    
    (
        lambda x: LGBMRegressor(
            random_state=42,
            verbosity=-1,
        ),
        {
            'n_estimators': [50, 100, 200],  # Number of trees in the ensemble
            'max_depth': [4, 5, 10],  # Max depth of each tree
            'learning_rate': [0.05, 0.1, 0.2]  # Learning rate
        },
        dict(),
    ),
    
    (
        lambda x: StackingRegressor(
            estimators=[
                ('dt', DecisionTreeRegressor(max_depth=5, random_state=42)),
                ('bagging', BaggingRegressor(
                    base_estimator=DecisionTreeRegressor(max_depth=5, random_state=42),
                    n_estimators=100, random_state=42)
                )
            ],
            final_estimator=Ridge()
        ),
        {
            'final_estimator__alpha': [0.1, 1.0, 10.0],  # Regularization strength for Ridge meta-model
        },
        dict(),
    ),
    
    (
        lambda x: BayesianRidge(),
        {
            'n_iter': [100, 200, 300],  # Number of iterations
            'alpha_1': [1e-6, 1e-5, 1e-4],  # Hyperparameter for the weight of the prior for alpha
            'alpha_2': [1e-6, 1e-5, 1e-4],  # Hyperparameter for the weight of the prior for beta
            'lambda_1': [1e-6, 1e-5, 1e-4],  # Hyperparameter for the weight of the prior for lambda
            'lambda_2': [1e-6, 1e-5, 1e-4]  # Hyperparameter for the weight of the prior for lambda
        },
        dict(),
    ),
    
    (
        lambda x: MLPRegressor(
            random_state=0,
            activation="relu",
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
            'C': [1000, 100, 10, 1, 0.1],
            'gamma': [1e-5, 1e-4, 1e-3, 1e-2],
        },
        dict(),
    ),
    
    (
        lambda x: KNeighborsRegressor(),
        {
            "n_neighbors": list(range(1,31)),
        },
        dict(),
    ),
    
    (
        lambda x: RandomForestRegressor(),
        {
            'max_depth': [100, 110, 120, 130,140,150],
            'max_features': [4, 5, 6],
            'min_samples_leaf': [4, 5, 6, 8],
            'min_samples_split': [10, 12, 14, 16],
            'n_estimators': [200, 400, 600, 1000, 1200]
        },
        dict(),
    ),
    
    (
        lambda x: GradientBoostingRegressor(),
        {
            "n_estimators": [1, 2, 5, 10, 20, 50, 100, 200, 500,1000,2000],
            "max_leaf_nodes": [2, 5, 10, 20, 50, 100],
            "learning_rate": [1, 0.1, 0.01, 0.001],
            'max_depth':[1,2,4],
            'subsample':[.5,.75,1]
        },
        dict(),
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
    CONFIG_sat,
    CONFIG_sat_clim,
    CONFIG_sat_chm,
    CONFIG_sat_clim_chm
]
TARGETS = [
    TARGET_CARBON_STORAGE,
    TARGET_CARBON_SEQUESTRATION,
]