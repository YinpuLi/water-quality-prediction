import pandas as pd
import os
import sys
import ast
from typing import Union

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)


def hyperparamter_tuning(
        X_train: pd.DataFrame
        , y_train: pd.DataFrame
        , param_grid: dict
        , model: Union[]
        , 
)