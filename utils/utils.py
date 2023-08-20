import pandas as pd
from scipy.io import loadmat
import os
import sys

root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root_dir)

def get_absolute_path(
          file_name:str='water_dataset.mat'
          , rel_path:str='data'
          , base_dir = '/Users/yinpuli/Documents/python-projects/water-quality-prediction'#os.path.abspath(os.path.join(os.getcwd(), '..'))
):
     return os.path.join(base_dir, rel_path, file_name)


def func():
     return 



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

