import pandas as pd
from scipy.io import loadmat
from datetime import datetime, timedelta
import joblib
import os
import sys


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)



from utils.constants import *



# def save_model(file_name, model, model_info=None):
#     """
#     Save a model and its associated info (if provided) to a file using joblib.

#     Parameters:
#     file_name (str): The name of the file to save the model.
#     model (object): The model object to be saved.
#     model_info (dict, optional): Additional information about the model (default: None).

#     # Usage example
#     best_model = xgb.XGBRegressor(objective='reg:squarederror')
#     best_params = {'max_depth': 5, 'learning_rate': 0.01, 'n_estimators': 100}
#     best_score = 0.85
#     model_info = {'best_params': best_params, 'best_score': best_score}

#     save_model('best_xgb_model.joblib', best_model, model_info)

#     """
#     directory = os.path.dirname(file_name)
#     os.makedirs(directory, exist_ok=True)
#     joblib.dump((model, model_info), file_name)


def save_model(file_name, model, model_info=None):
    """
    Save a model and its associated info (if provided) to a file using joblib.

    Parameters:
    file_name (str): The name of the file to save the model.
    model (object): The model object to be saved.
    model_info (dict, optional): Additional information about the model (default: None).
    """
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    # elif os.path.exists(file_name):
    #     raise ValueError(f"File '{file_name}' already exists. Please provide a new file name.")

    joblib.dump((model, model_info), file_name)



def load_model(file_name):
    """
    Load a saved model and its associated info using joblib.

    Parameters:
    file_name (str): The name of the file containing the saved model.

    Returns:
    model: The loaded model object.
    model_info: Additional information about the model (if available).
    
    # Load the model and its info
    loaded_model, loaded_model_info = load_model(best_xgb_file)
    """
    return joblib.load(file_name)



# Now you can use the loaded_model and loaded_model_info variables
# to access the loaded XGBoost model and its associated info.



def get_train_dates_col(start_date: datetime = datetime.strptime(train_start, "%Y-%m-%d"),
                  num_days: int = n_tr) -> list:
    """
    Generate a list of date strings starting from a given start_date.

    Args:
        start_date (datetime, optional): The start date. Defaults to "2016-01-28".
        num_days (int, optional): The number of days to generate. Defaults to 423.

    Returns:
        list: A list of date strings in the format "%Y-%m-%d".


    Example usage
    print(get_train_dates_col(start_date=datetime.strptime(train_start, "%Y-%m-%d"), num_days=1))
    # Expected results: 
    ['2016-01-28']
    """
    date_strings = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(num_days)]
    return date_strings



def get_test_dates_col(end_date: datetime = datetime.strptime(test_end, "%Y-%m-%d"),#datetime.today(),
                  num_days: int = n_te) -> list:
    """
    Generate a list of date strings up to a given end_date.

    Args:
        end_date (datetime, optional): The end date. Defaults to today's date.
        num_days (int, optional): The number of days to generate. Defaults to 423.

    Returns:
        list: A list of date strings in the format "%Y-%m-%d".

    Example usage:
    print(get_test_dates_col(num_days=1)) # ['2018-01-01']
    """
    start_date = end_date - timedelta(days=num_days - 1)
    date_strings = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(num_days)]
    return date_strings




def add_temporal_spatial_cols(
        loaded_data
        , start_date=None
        , end_date=None
        , train_num_days=None
        , test_num_days=None
):
    """
    Adding new feature columns to train/test data sets,
    which contain time (contiguous dates), location (location_ids) info.

    loaded_data: raw results (a dictionary) from load_data
    start_date: the start date for the date column
    num_days: the number of days for which the date column needs to be added
    """
    if start_date is None:
        start_date = datetime.strptime(train_start, "%Y-%m-%d")
    if end_date is None:
        end_date = datetime.strptime(test_end, "%Y-%m-%d")
    if train_num_days is None:
        train_num_days = len(loaded_data['X_tr'][0])
    if test_num_days is None: 
        test_num_days = len(loaded_data['X_te'][0])

    # Extract the individual column names from the nested array
    features = [name[0] for name in loaded_data['features'][0]]

    # Convert the NumPy arrays to Pandas DataFrames
    train_dfs = [pd.DataFrame(arr, columns=features) for arr in loaded_data['X_tr'][0]]
    test_dfs = [pd.DataFrame(arr, columns=features) for arr in loaded_data['X_te'][0]]

    # Iterate through the X_tr data frames and add the date and location columns
    for d, df in enumerate(train_dfs):
        date = start_date + timedelta(days=d)
        date_str = date.strftime("%Y-%m-%d")

        for index, row in df.iterrows():
            location_id = loaded_data['location_ids'][index]
            df.loc[index, 'Date'] = date_str
            df.loc[index, 'Location_ID'] = location_id
            # df.loc[index, 'Year']    = str(int(date.year))
            df.loc[index, 'Month']   = str(int(date.month))
            df.loc[index, 'Week']    = str(int(date.isocalendar().week))
            df.loc[index, 'Weekday'] = str(int(date.weekday()))

            # Add season based on month
            if 3 <= date.month <= 5:
                df.loc[index, 'Season'] = 'Spring'
            elif 6 <= date.month <= 8:
                df.loc[index, 'Season'] = 'Summer'
            elif 9 <= date.month <= 11:
                df.loc[index, 'Season'] = 'Fall'
            else:
                df.loc[index, 'Season'] = 'Winter'


    # Iterate through the X_te data frames and add the date and location columns
    for d, df in reversed(list(enumerate(test_dfs))):
        date = end_date - timedelta(days=d)
        date_str = date.strftime("%Y-%m-%d")

        for index, row in df.iterrows():
            location_id = loaded_data['location_ids'][index]
            df.loc[index, 'Date'] = date_str
            df.loc[index, 'Location_ID'] = location_id
            # df.loc[index, 'Year']    = str(int(date.year))
            df.loc[index, 'Month']   = str(int(date.month))
            df.loc[index, 'Week']    = str(int(date.isocalendar().week))
            df.loc[index, 'Weekday'] = str(int(date.weekday()))

            # Add season based on month
            if 3 <= date.month <= 5:
                df.loc[index, 'Season'] = 'Spring'
                df.loc[index, 'Season_Num'] = 1  # Spring
            elif 6 <= date.month <= 8:
                df.loc[index, 'Season'] = 'Summer'
                df.loc[index, 'Season_Num'] = 2  # Summer
            elif 9 <= date.month <= 11:
                df.loc[index, 'Season'] = 'Fall'
                df.loc[index, 'Season_Num'] = 3  # Fall
            else:
                df.loc[index, 'Season'] = 'Winter'
                df.loc[index, 'Season_Num'] = 4  # Winter


    # Store the modified DataFrames back into the dictionary
    loaded_data['X_tr'][0] = [df.values for df in train_dfs]
    loaded_data['X_te'][0] = [df.values for df in test_dfs]


    return loaded_data










def stack_dataframes(loaded_data):
    # Stack X_tr dataframes by row
    stacked_X_tr = pd.concat([pd.DataFrame(arr) for arr in loaded_data['X_tr'][0]], ignore_index=True)

    # Stack X_te dataframes by row
    stacked_X_te = pd.concat([pd.DataFrame(arr) for arr in loaded_data['X_te'][0]], ignore_index=True)

    return stacked_X_tr, stacked_X_te




def get_absolute_path(
          file_name:str='water_dataset.mat'
          , rel_path:str='data'
          , base_dir = '/Users/yinpuli/Documents/python-projects/water-quality-prediction'#os.path.abspath(os.path.join(os.getcwd(), '..'))
):
     return os.path.join(base_dir, rel_path, file_name)



def load_data(file_path#='data/water_dataset.mat'
              ):
    loaded_data = loadmat(file_path)
    return loaded_data
    



########## input: for training set 


def data_prep_X(df):
    
    """
    Prepare train/test feature matrix, flatten the time-spatial into rows.

    df_out 
        - for train is  (37 * 423) * 11
        - for test is (37 * 283) * 11

    n_rows = (37 * 423)
        - K = 37 locations
        - N_{tr} = 423 days

    n_col = 11
        - p = 11 features
    
    """

    df_out = pd.DataFrame()
    
    length_df = df.shape[1]

    for i in range(length_df):

            df_out = pd.concat([df_out,pd.DataFrame(df[0,i])], axis = 0)
    
    
    return df_out


def gen_col_name(p=11):
     # Generating a list of column names from X1 to X10
    column_names = [f'X{i}' for i in range(1, p+1)]
    return column_names



def data_pre_Y(df):
    
    """
    Prepare train/test response set, flatten the time-spatial into rows.

    df_out 
        - for train is  (37 * 423) * 1
        - for test is (37 * 283) * 1

    n_rows = (37 * 423)
        - K = 37 locations
        - N_{tr} = 423 days

    n_col = 11
        - p = 11 features
    
    """
    df = pd.DataFrame(df)
    df_out = pd.DataFrame()
    
    length_df = df.shape[1]

    
    for i in range(length_df):
        df_out = pd.concat([df_out,df.loc[:,i]], axis = 0)
    
    
    return df_out




# def get_df():



def save_csv(df, file_path, index=False):
    _dir = os.path.dirname(file_path)
    os.makedirs(_dir, exist_ok=True)

    df.to_csv(file_path, index=index)
