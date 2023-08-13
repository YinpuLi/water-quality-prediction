import pandas as pd
import os
import sys
import ast

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.utils import load_data, save_csv
from utils.constants import *

data = load_data('data/water_dataset.mat')




# # HAVE NO IDEA
# # Convert string representation of arrays to actual arrays
# for key in ['X_tr', 'X_te', 'Y_tr', 'Y_te']:
#     data[key] = ast.literal_eval(data[key])

# # Create a list of dictionaries containing data and feature names
# x_tr_data = []
# for array in data['X_tr']:
#     x_tr_data.append({'feature_names': data['features'], 'data': array})

# # Create a DataFrame from the list of dictionaries
# x_tr_df = pd.DataFrame(x_tr_data)

# print(x_tr_df)

# # # Save the DataFrame to a CSV file
# # x_tr_df.to_csv('x_tr.csv', index=False)

# # save_csv