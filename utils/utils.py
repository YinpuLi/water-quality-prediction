import pandas as pd
from scipy.io import loadmat
from datetime import datetime, timedelta

import os
import sys


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)



from utils.constants import *




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








def get_absolute_path(
          file_name:str='water_dataset.mat'
          , rel_path:str='data'
          , base_dir = '/Users/yinpuli/Documents/python-projects/water-quality-prediction'#os.path.abspath(os.path.join(os.getcwd(), '..'))
):
     return os.path.join(base_dir, rel_path, file_name)



def load_data(file_path#='data/water_dataset.mat'
              ):
    loaded_data = loadmat(file_path)
    # loaded_df = pd.json_normalize(loaded_data)
    # return loaded_df
    return loaded_data
    
# TODO: need to decide what to do formulate the data.


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
