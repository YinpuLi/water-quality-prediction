import pandas as pd
import numpy as np
import os
import sys
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.utils     import *
from utils.constants import *
from utils.metrics   import *
from src.model_tuning import *


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

scoring=make_scorer(lambda y_true, y_pred: -mean_squared_error(y_true, y_pred, squared=False))

######### XGBoost #########

best_xgb_file = get_absolute_path(
    file_name = 'best_xgb_model.joblib'
    , rel_path = 'results'
)

xgb_result = hyperparameter_tuning(
        X_train=X_train_preprocessed_df,
        y_train=y_train,
        X_test=X_test_preprocessed_df,
        y_test=y_test,
        param_grid={
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01],
            'n_estimators': [100, 200, 500],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1, 0.5],
            'min_child_weight': [1, 5, 10, 20],
            'random_state': [RANDOM_SEED]
        },
        model=xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_SEED),
        scoring=scoring,
        eval_func=compute_metrics,
        file_path=best_xgb_file,
        cv=5
) 

print("Success of xgboost!")

######### Random Forest #########

best_rf_file = get_absolute_path(
    file_name = 'best_rf_model.joblib'
    , rel_path = 'results'
)

rf_result = hyperparameter_tuning(
        X_train=X_train_preprocessed_df,
        y_train=y_train,
        X_test=X_test_preprocessed_df,
        y_test=y_test,
        param_grid={
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [None, 3, 5, 10, 20],
            'min_samples_split': [1, 2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'random_state': [RANDOM_SEED]
        },
        model=RandomForestRegressor(random_state=RANDOM_SEED),
        scoring=scoring,
        eval_func=compute_metrics,
        file_path=best_rf_file,
        cv=5
) 

print("Success of rf!")



######### MLP #########

best_mlp_file = get_absolute_path(
    file_name = 'best_mlp_model.joblib'
    , rel_path = 'results'
)

mlp_result = hyperparameter_tuning(
        X_train=X_train_preprocessed,
        y_train=y_train,
        X_test=X_test_preprocessed,
        y_test=y_test,
        param_grid= {
            'hidden_layer_sizes': list(zip([1000], [300])),
            'max_iter': [50, 100],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive'],
            'random_state': [RANDOM_SEED]
        },
        model=MLPRegressor(random_state=RANDOM_SEED, hidden_layer_sizes = (20, 20)),
        scoring="neg_root_mean_squared_error",
        eval_func=compute_metrics,
        file_path=best_mlp_file,
        cv=5
) 

print("Success of MLP!")


######### ElasticNet #########

best_lin_file = get_absolute_path(
    file_name = 'best_lin_model.joblib'
    , rel_path = 'results'
)

lin_result = hyperparameter_tuning(
        X_train=X_train_preprocessed_df,
        y_train=y_train,
        X_test=X_test_preprocessed_df,
        y_test=y_test,
        param_grid={
            "alpha":[0.01, 0.001, 0.0001, 0.00001, 0.000001],
            "l1_ratio": [0, 0.1, 0.3, 0.5, 0.8, 1],
            'random_state': [RANDOM_SEED]
        },
        model=ElasticNet(random_state=RANDOM_SEED, l1_ratio=1).fit(X_train_preprocessed_df,y_train),
        scoring="neg_root_mean_squared_error",
        eval_func=compute_metrics,
        file_path=best_lin_file,
        cv=5
) 

print("Success of ElasticNet!")
