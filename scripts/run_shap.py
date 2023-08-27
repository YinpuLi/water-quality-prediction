import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error,mean_absolute_error

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.utils import *
from utils.constants import *
from src.shap import *


######## Import data ########


y_train = pd.read_csv(get_absolute_path('y_train.csv', 'data'))
y_test = pd.read_csv(get_absolute_path('y_test.csv', 'data'))


stack_train = pd.read_csv(get_absolute_path('stacked_X_tr.csv', 'data'))
stack_test  = pd.read_csv(get_absolute_path('stacked_X_te.csv', 'data'))

stack_train = stack_train.astype(column_data_extended_types)
stack_test = stack_test.astype(column_data_extended_types)



######## Feature Engineering ##########

# Select numeric and categorical columns
numeric_columns = stack_train.select_dtypes(include=['float64']).columns
categorical_columns = [#'Date', 
                       'Location_ID',
                    #    'Year',
                       'Month',
                       'Week',
                       'Weekday',
                       'Season'
                       ]  # Add any categorical columns here

# Create preprocessing transformers
numeric_transformer = StandardScaler()  # we can use other scalers as well
categorical_transformer = OneHotEncoder(drop=None)  # Use one-hot encoding for categorical columns

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

# Fit the preprocessor on training data and transform both train and test data
X_train_preprocessed = preprocessor.fit_transform(stack_train)
X_test_preprocessed  = preprocessor.transform(stack_test)


# Get the column names after one-hot encoding
categorical_encoded_columns = preprocessor.named_transformers_['cat']\
                                    .get_feature_names_out(input_features=categorical_columns)

# Convert X_train_preprocessed and X_test_preprocessed to DataFrames

X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed.toarray(), columns=np.concatenate([numeric_columns, categorical_encoded_columns]))
X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed.toarray(), columns=np.concatenate([numeric_columns, categorical_encoded_columns]))


########### Generate SHAP results


# List of model names
model_names = ['xgb', 'rf' , #'mlp', 
               'lin'
]

# Relative path
rel_path = 'results'

# # Dictionary to store file paths for each model
# model_file_paths = {}

# Loop through each model name
for model_name in model_names:
    file_paths = generate_file_paths_for_shap(model_name, rel_path)
    # model_file_paths[model_name] = file_paths
    print(f"{model_name.capitalize()} Model File Paths:", file_paths)

    gen_shap_results(
        load_file_path = file_paths[0]
        , save_file_path_1 = file_paths[1]
        , save_file_path_2 = file_paths[2]
        , refit_X = X_train_preprocessed_df
        , refit_y = y_train
        , figure_dpi = 300
    )

########### Generate SHAP results on test set


# List of model names
model_names = ['xgb', 'rf' , #'mlp', 
               'lin'
]

# Relative path
rel_path = 'results' 

# # Dictionary to store file paths for each model
# model_file_paths = {}

# Loop through each model name
for model_name in model_names:
    file_paths = generate_file_paths_for_shap_2(model_name, rel_path)
    # model_file_paths[model_name] = file_paths
    print(f"{model_name.capitalize()} Model File Paths:", file_paths)

    gen_shap_results(
        load_file_path = file_paths[0]
        , save_file_path_1 = file_paths[1]
        , save_file_path_2 = file_paths[2]
        , refit_X = X_test_preprocessed_df
        , refit_y = y_test
        , figure_dpi = 300
    )


