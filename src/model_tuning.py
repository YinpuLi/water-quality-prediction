import pandas as pd
import os
import sys
import ast
from typing import Union
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.utils     import *
from utils.constants import *
from utils.metrics   import *

# Scripts:


def hyperparameter_tuning(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        param_grid: dict,
        model: Union[xgb.sklearn.XGBRegressor, RandomForestRegressor
                     , lgb.LGBMRegressor, MLPRegressor
                     ],
        scoring,
        eval_func,
        file_path: str,
        cv: int
) -> dict:
    """
    Perform hyperparameter tuning, prediction, and evaluation for a given model.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.DataFrame): Training target values.
    X_test (pd.DataFrame): Testing features.
    y_test (pd.DataFrame): Testing target values.
    param_grid (dict): Grid of hyperparameters for tuning.
    model (Union[xgb.sklearn.XGBRegressor, RandomForestRegressor]): Initialized model.
    scoring (function): Scoring function for hyperparameter tuning.
    eval_func (function): Evaluation function for model performance.
    file_path (str): File path for saving the best model and results.
    cv (int): Number of cross-validation folds.

    Returns:
    dict: Dictionary containing best model information and evaluation metrics.
    
    Example:
    X_train = X_train_preprocessed_df
    y_train = y_train
    X_test  = X_test_preprocessed
    y_test = y_test
    model = xgb.XGBRegressor(objective='reg:squarederror') # RandomForestRegressor()
    scoring = make_scorer(lambda y_true, y_pred: -mean_squared_error(y_true, y_pred, squared=False))
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 3, 5, 10, 20],
        'min_samples_split': [1, 2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    eval_func = compute_metrics
    
    """

    # Perform hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=6, verbose=1)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_

    # Making predictions on the validation data using the best model
    y_pred = best_model.predict(X_test)

    # Calculate evaluation metrics
    _df = pd.DataFrame(
        {
            'forecast': y_pred,
            'actual': y_test['measurement']
        }
    )
    eval_metrics = eval_func(_df, EVAL_METRIC_LIST)

    # Get feature importance scores for tree based models

    if isinstance(best_model, MLPRegressor): # type(best_model) == MLPRegressor
        feature_importance_dict = {}
    else:
        # Get feature importance scores
        feature_importance = best_model.feature_importances_
        feature_names = X_train.columns
        feature_importance_dict = dict(zip(feature_names, feature_importance))

    # Save the results
    best_model_info = {
        'best_params': best_params,
        'best_score': best_score,
        'feature_importance': feature_importance_dict,
        'y_pred': y_pred,
        'eval_metrics': eval_metrics
    }
    
    # Save the best model and results
    save_model(file_name=file_path, model=best_model, model_info=best_model_info)

    return best_model_info


