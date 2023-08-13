import pandas as pd
from scipy.io import loadmat
import os

def load_data(file_path='data/water_dataset.mat'):
    loaded_data = loadmat(file_path)
    # loaded_df = pd.json_normalize(loaded_data)
    # return loaded_df
    return loaded_data
    
# TODO: need to decide what to do formulate the data.


# def get_df():



def save_csv(df, file_path, index=False):
    _dir = os.path.dirname(file_path)
    os.makedir(_dir, exit_ok=True)

    df.to_csv(file_path, index = index)