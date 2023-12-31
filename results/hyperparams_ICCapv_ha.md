+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| config   | model                     | hyperparams             | R2           | RMSE        | %RMSE        |
+==========+===========================+=========================+==============+=============+==============+
| CONFIG_0 | CatBoostRegressor         | learning_rate: 0.03     | 0.27 ± 0.09  | 1.01 ± 0.10 | 59.26 ± 7.41 |
|          |                           | l2_leaf_reg: 0.5        |              |             |              |
|          |                           | iterations: 150         |              |             |              |
|          |                           | depth: 2                |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_0 | GradientBoostingRegressor | subsample: 0.75         | 0.14 ± 0.13  | 1.09 ± 0.08 | 64.22 ± 5.95 |
|          |                           | n_estimators: 100       |              |             |              |
|          |                           | max_leaf_nodes: 100     |              |             |              |
|          |                           | max_depth: 2            |              |             |              |
|          |                           | learning_rate: 0.1      |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_0 | KNeighborsRegressor       | n_neighbors: 16         | -0.03 ± 0.12 | 1.19 ± 0.06 | 70.13 ± 5.71 |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_0 | LinearRegression          |                         | 0.13 ± 0.13  | 1.09 ± 0.06 | 64.34 ± 5.25 |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_0 | MLPRegressor              | max_iter: 20            | -0.08 ± 0.15 | 1.22 ± 0.13 | 72.18 ± 9.45 |
|          |                           | learning_rate_init: 0.1 |              |             |              |
|          |                           | hidden_layer_sizes: [8] |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_0 | RandomForestRegressor     | n_estimators: 200       | 0.29 ± 0.09  | 1.00 ± 0.09 | 58.67 ± 7.19 |
|          |                           | min_samples_split: 16   |              |             |              |
|          |                           | min_samples_leaf: 5     |              |             |              |
|          |                           | max_features: 5         |              |             |              |
|          |                           | max_depth: 130          |              |             |              |
|          |                           | bootstrap: True         |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_0 | SVR                       | gamma: 1e-05            | -0.05 ± 0.09 | 1.21 ± 0.09 | 71.17 ± 7.37 |
|          |                           | C: 1                    |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_0 | XGBRegressor              | n_estimators: 700       | 0.18 ± 0.13  | 1.06 ± 0.09 | 62.46 ± 6.55 |
|          |                           | max_depth: 4            |              |             |              |
|          |                           | learning_rate: 0.01     |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_1 | CatBoostRegressor         | learning_rate: 0.1      | 0.40 ± 0.08  | 0.91 ± 0.09 | 53.60 ± 6.75 |
|          |                           | l2_leaf_reg: 3          |              |             |              |
|          |                           | iterations: 200         |              |             |              |
|          |                           | depth: 6                |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_1 | GradientBoostingRegressor | subsample: 1            | 0.31 ± 0.13  | 0.98 ± 0.11 | 57.48 ± 7.45 |
|          |                           | n_estimators: 1000      |              |             |              |
|          |                           | max_leaf_nodes: 100     |              |             |              |
|          |                           | max_depth: 4            |              |             |              |
|          |                           | learning_rate: 0.01     |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_1 | KNeighborsRegressor       | n_neighbors: 14         | 0.06 ± 0.08  | 1.14 ± 0.08 | 67.34 ± 6.67 |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_1 | LinearRegression          |                         | 0.32 ± 0.11  | 0.97 ± 0.09 | 57.11 ± 6.37 |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_1 | MLPRegressor              | max_iter: 50            | -0.03 ± 0.04 | 1.20 ± 0.09 | 70.59 ± 7.57 |
|          |                           | learning_rate_init: 0.1 |              |             |              |
|          |                           | hidden_layer_sizes: [2] |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_1 | RandomForestRegressor     | n_estimators: 1000      | 0.41 ± 0.07  | 0.90 ± 0.10 | 53.33 ± 7.41 |
|          |                           | min_samples_split: 10   |              |             |              |
|          |                           | min_samples_leaf: 5     |              |             |              |
|          |                           | max_features: 6         |              |             |              |
|          |                           | max_depth: 150          |              |             |              |
|          |                           | bootstrap: True         |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_1 | SVR                       | gamma: 1e-05            | 0.06 ± 0.07  | 1.14 ± 0.08 | 67.46 ± 6.96 |
|          |                           | C: 1                    |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_1 | XGBRegressor              | n_estimators: 600       | 0.31 ± 0.10  | 0.98 ± 0.09 | 57.56 ± 6.72 |
|          |                           | max_depth: 5            |              |             |              |
|          |                           | learning_rate: 0.02     |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_2 | CatBoostRegressor         | learning_rate: 0.03     | 0.42 ± 0.09  | 0.90 ± 0.11 | 52.88 ± 8.07 |
|          |                           | l2_leaf_reg: 0.5        |              |             |              |
|          |                           | iterations: 150         |              |             |              |
|          |                           | depth: 2                |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_2 | GradientBoostingRegressor | subsample: 1            | 0.31 ± 0.10  | 0.97 ± 0.10 | 57.33 ± 6.90 |
|          |                           | n_estimators: 1000      |              |             |              |
|          |                           | max_leaf_nodes: 100     |              |             |              |
|          |                           | max_depth: 4            |              |             |              |
|          |                           | learning_rate: 0.01     |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_2 | KNeighborsRegressor       | n_neighbors: 16         | -0.02 ± 0.12 | 1.19 ± 0.07 | 69.96 ± 5.76 |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_2 | LinearRegression          |                         | 0.30 ± 0.09  | 0.98 ± 0.08 | 57.89 ± 6.28 |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_2 | MLPRegressor              | max_iter: 50            | -0.04 ± 0.05 | 1.20 ± 0.09 | 70.70 ± 7.35 |
|          |                           | learning_rate_init: 0.1 |              |             |              |
|          |                           | hidden_layer_sizes: [2] |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_2 | RandomForestRegressor     | n_estimators: 1000      | 0.42 ± 0.08  | 0.90 ± 0.10 | 52.93 ± 7.47 |
|          |                           | min_samples_split: 10   |              |             |              |
|          |                           | min_samples_leaf: 5     |              |             |              |
|          |                           | max_features: 6         |              |             |              |
|          |                           | max_depth: 150          |              |             |              |
|          |                           | bootstrap: True         |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_2 | SVR                       | gamma: 1e-05            | 0.02 ± 0.13  | 1.16 ± 0.08 | 68.59 ± 6.30 |
|          |                           | C: 10                   |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_2 | XGBRegressor              | n_estimators: 700       | 0.32 ± 0.11  | 0.97 ± 0.10 | 56.98 ± 7.16 |
|          |                           | max_depth: 4            |              |             |              |
|          |                           | learning_rate: 0.01     |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_3 | CatBoostRegressor         | learning_rate: 0.1      | 0.23 ± 0.11  | 1.03 ± 0.10 | 60.66 ± 7.19 |
|          |                           | l2_leaf_reg: 1          |              |             |              |
|          |                           | iterations: 100         |              |             |              |
|          |                           | depth: 2                |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_3 | GradientBoostingRegressor | subsample: 0.5          | 0.22 ± 0.06  | 1.04 ± 0.09 | 61.37 ± 6.88 |
|          |                           | n_estimators: 100       |              |             |              |
|          |                           | max_leaf_nodes: 20      |              |             |              |
|          |                           | max_depth: 4            |              |             |              |
|          |                           | learning_rate: 0.01     |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_3 | KNeighborsRegressor       | n_neighbors: 14         | 0.06 ± 0.08  | 1.14 ± 0.08 | 67.41 ± 6.68 |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_3 | LinearRegression          |                         | 0.16 ± 0.14  | 1.08 ± 0.07 | 63.39 ± 5.43 |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_3 | MLPRegressor              | max_iter: 50            | -0.03 ± 0.05 | 1.20 ± 0.09 | 70.67 ± 7.32 |
|          |                           | learning_rate_init: 0.1 |              |             |              |
|          |                           | hidden_layer_sizes: [2] |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_3 | RandomForestRegressor     | n_estimators: 600       | 0.28 ± 0.08  | 1.00 ± 0.10 | 59.07 ± 7.37 |
|          |                           | min_samples_split: 16   |              |             |              |
|          |                           | min_samples_leaf: 5     |              |             |              |
|          |                           | max_features: 4         |              |             |              |
|          |                           | max_depth: 110          |              |             |              |
|          |                           | bootstrap: True         |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_3 | SVR                       | gamma: 0.0001           | -0.02 ± 0.13 | 1.19 ± 0.08 | 69.87 ± 6.48 |
|          |                           | C: 10                   |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+
| CONFIG_3 | XGBRegressor              | n_estimators: 700       | 0.18 ± 0.12  | 1.06 ± 0.07 | 62.67 ± 5.87 |
|          |                           | max_depth: 4            |              |             |              |
|          |                           | learning_rate: 0.01     |              |             |              |
+----------+---------------------------+-------------------------+--------------+-------------+--------------+