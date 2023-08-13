import pandas as pd
from scipy.io import loadmat

def load_data(file_path='data/water_dataset.mat'):
    loaded_data = loadmat(file_path)
    # loaded_df = pd.json_normalize(loaded_data)

    # return loaded_df
    return loaded_data
    