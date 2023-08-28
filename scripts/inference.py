import pandas as pd
import os
import sys
import shap

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.utils     import *
from utils.constants import *
from utils.metrics   import *
from src.shap import *

# List of model names
model_names = ['xgb', 'rf' , 
               'mlp',
                 'lin'
]

# Relative path
rel_path = 'results' 

# Create empty lists to store evaluation metrics and best scores/feature importance
eval_metrics_list = []
best_scores_list = []

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

# Create a DataFrame for evaluation metrics
eval_metrics_df = pd.DataFrame(eval_metrics_list)
eval_metrics_df.insert(0, 'Model', model_names)
print("Summary Table for Evaluation Metrics:")
print(eval_metrics_df)

# Create separate DataFrames for best scores and feature importance
best_scores_df = pd.DataFrame({"Model": model_names, "Best Score": best_scores_list})


best_model_comparison_best_scores_path        = get_absolute_path(
    file_name='best_scores.csv'
    , rel_path='results' + '/' + 'summary'
)     
best_model_comparison_eval_metrics_path       = get_absolute_path(
    file_name='eval_metrics.csv'
    , rel_path='results' + '/' + 'summary'
)   


best_scores_df.to_csv(best_model_comparison_best_scores_path, index = True)
eval_metrics_df.to_csv(best_model_comparison_eval_metrics_path, index = True)
