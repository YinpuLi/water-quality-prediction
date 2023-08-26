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
import matplotlib.pyplot as plt

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.utils     import *
from utils.constants import *
from utils.metrics   import *


def gen_shap_results(
    load_file_path: str
    , save_file_path_1: str # for bar plot, showing feature importance
    , save_file_path_2: str # for SHAP value plot
    , refit_X: pd.DataFrame
    , refit_y: pd.DataFrame
    # , bbox_inches=None
    # , pad_inches=0.1
    # , facecolor='auto'
    # , edgecolor='auto',
    # , backend=None
    , figure_dpi: int
    , dpi = 'figure'

    
):

    """

    load_file_path: Path to the saved model file
    save_file_path_1: Path to save SHAP bar plot
    save_file_path_2: Path to save SHAP value plot
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
    best_model, best_model_info = load_model(load_file_path)

    
    # Get the best hyperparameters and best model
    best_params = best_model_info['best_params']
    # y_pred      = best_model_info['y_pred']
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
        shap_values = explainer.shap_values(refit_X)
        shap.summary_plot(shap_values, refit_X, plot_type = 'bar')
        plt.gcf().set_size_inches(15, 10)
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_file_path_1), exist_ok=True)
        
        plt.savefig(save_file_path_1, dpi=figure_dpi)
        plt.show()
        
            

        shap.summary_plot(shap_values, refit_X)
        plt.gcf().set_size_inches(15, 10)
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_file_path_2), exist_ok=True)
        
        plt.savefig(save_file_path_2, dpi=figure_dpi)
        plt.show()
        

    else:
        print("Not implemented yet...")
        return 
        



# ## XGBoost
# best_xgb_file = get_absolute_path(
#     file_name = 'best_xgb_model.joblib'
#     , rel_path = 'results'
# )

# best_xgb_shap_file_1 = get_absolute_path(
#     file_name = 'best_xgb_shap_bar.png'
#     , rel_path = 'results' + '/' + 'shap'
# )


# best_xgb_shap_file_2 = get_absolute_path(
#     file_name = 'best_xgb_shap_val.png'
#     , rel_path = 'results' + '/' + 'shap'
# )

# gen_shap_results(
#     load_file_path = best_xgb_file
#     , save_file_path_1 = best_xgb_shap_file_1
#     , save_file_path_2 = best_xgb_shap_file_2
#     , refit_X = X_train_preprocessed_df
#     , refit_y = y_train
#     , figure_dpi = 300
# )