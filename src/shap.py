import pandas as pd
import shap
import os
import sys
import ast
from typing import Union
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import shap

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.utils     import *
from utils.constants import *
from utils.metrics   import *


def gen_shap_results(
    load_file_path: str
    , save_file_path: str
    , refit_X: pd.DataFrame
    , refit_y: pd.DataFrame
    , figure_dpi: int
    
):

    """

    load_file_path: Path to the saved model file
    save_file_path: Path to save SHAP plots
    refit_X: DataFrame containing input features for refitting
    refit_y: DataFrame containing target values for refitting
    figure_dpi: DPI value for saving figures

    # Example: 
    load_file_path = best_xgb_file = get_absolute_path(
                        file_name = 'best_xgb_model.joblib'
                        , rel_path = 'results'
                    )
    save_file_path = best_xgb_shap_file = get_absolute_path(
                        file_name = 'best_xgb_shap.png'
                        , rel_path = 'results' + '/' + 'shap'
                    )
    refit_X = X_train_preprocessed_df # X_test_preprocessed_df
    refit_y = y_train # y_test 
    # TODO: ask Mao, should we use train or test?

    figure_dpi = 300
    """

    # Load the best model and its info file
    best_model, best_model_info = load_model(file_path)

    
    # Get the best hyperparameters and best model
    best_params = best_model_info['best_params']
    y_pred      = best_model_info['y_pred']
    eval_metrics= best_model_info['eval_metrics']

    # Refit model
    refit_model = best_model.fit(refit_X, refit_y)

    # Using SHAP
    np.bool = np.bool_
    np.int = np.int_

    ## Tree based methods are handled 
    if isinstance(best_model, (xgb.XGBRegressor, RandomForestRegressor)):
        print("Tree Based Model...")
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_value(refit_X)
        shap.summary_plot(shap_values, refit_X, plot_type = 'bar')
        plt.tight_layout()  # Ensure plots are not cut off
        save_figure('shap_bar_plot', plt.gcf(), dpi=figure_dpi, save_location=save_file_path)
        
        shap.summary_plot(shap_values, refit_X)

    else:
        print("Not implemented yet...")
        



## XGBoost
best_xgb_file = get_absolute_path(
    file_name = 'best_xgb_model.joblib'
    , rel_path = 'results'
)

best_xgb_shap_file = get_absolute_path(
    file_name = 'best_xgb_shap.png'
    , rel_path = 'results' + '/' + 'shap'
)

