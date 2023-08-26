import pandas as pd
import os
import sys
import ast
from typing import Union
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.utils     import *
from utils.constants import *
from utils.metrics   import *


def hyperparamter_tuning(
        X_train: pd.DataFrame
        , y_train: pd.DataFrame
        , param_grid: dict
        , model: Union[xgb.sklearn.XGBRegressor]
        , scores_func: function

) -> dict:
    


    return {}