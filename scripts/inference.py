import pandas as pd
import shap
import os
import sys
import ast
from typing import Union
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import shap
import matplotlib.pyplot as plt

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.utils     import *
from utils.constants import *
from utils.metrics   import *
from src.shap import *

# List of model names
model_names = ['xgb', 'rf' , 
            #    'mlp',
                 'lin'
]

# Relative path
rel_path = 'results' 




# Create empty lists to store evaluation metrics and best scores/feature importance
eval_metrics_list = []
best_scores_list = []
feature_importance_list = []

# Loop through each model name
for model_name in model_names:
    file_paths = get_absolute_path(
        file_name='best_' + model_name + '_model.joblib',
        rel_path='results'
    )

    print(f"Best {model_name.capitalize()} Model File Paths:")
    print(file_paths)
    best_model, best_model_info = load_model(file_paths)

    # Append evaluation metrics, best scores, and feature importance to respective lists
    eval_metrics_list.append(best_model_info['eval_metrics'])
    best_scores_list.append(best_model_info['best_score'])
    feature_importance_list.append(best_model_info['feature_importance'])

# Create a DataFrame for evaluation metrics
eval_metrics_df = pd.DataFrame(eval_metrics_list)
eval_metrics_df.insert(0, 'Model', model_names)
print("Summary Table for Evaluation Metrics:")
print(eval_metrics_df)

# Create separate DataFrames for best scores and feature importance
best_scores_df = pd.DataFrame({"Model": model_names, "Best Score": best_scores_list})
# feature_importance_dfs = [pd.DataFrame({f"Feature Importance {i+1}": importance}) for i, importance in enumerate(feature_importance_list)]

# Create a list of DataFrames for feature importance
feature_importance_dfs = []
for i, importance in enumerate(feature_importance_list):
    df = pd.DataFrame({"Model": model_names, f"Feature Importance {i+1}": importance})
    feature_importance_dfs.append(df)
# Concatenate the feature importance DataFrames
feature_importance_combined = pd.concat(feature_importance_dfs, axis=0, ignore_index=True)


# # Concatenate the separate DataFrames horizontally to create the summary DataFrame
# summary_df = pd.concat([best_scores_df] + feature_importance_dfs, axis=1)
# print("Summary Table for Best Score and Feature Importance:")
# print(summary_df)

print("Feature Importance Table for Best Score and Feature Importance:")
print(feature_importance_dfs)


best_model_comparison_best_scores_path        = get_absolute_path(
    file_name='best_scores.csv'
    , rel_path='results' + '/' + 'summary'
)     
best_model_comparison_eval_metrics_path       = get_absolute_path(
    file_name='eval_metrics.csv'
    , rel_path='results' + '/' + 'summary'
)   
best_model_comparison_feature_importance_path = get_absolute_path(
    file_name='feature_importance.csv'
    , rel_path='results' + '/' + 'summary'
)

best_scores_df.to_csv(best_model_comparison_best_scores_path, index = True)
eval_metrics_df.to_csv(best_model_comparison_eval_metrics_path, index = True)
feature_importance_dfs.to_csv(best_model_comparison_feature_importance_path, index = True)
