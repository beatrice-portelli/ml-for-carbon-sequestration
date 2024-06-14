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


from configs import *


MODELS = [
    
    # format: (model_class, param_distributions, search_cv_args)
    
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