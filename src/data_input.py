import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from scipy.io import loadmat



loaded_data = loadmat('data/water_dataset.mat')

loaded_df = pd.json_normalize(loaded_data)

print(loaded_data.keys())
print()
print(loaded_df.columns)
# Index(['__header__', '__version__', '__globals__', 'X_tr', 'X_te', 'Y_tr',
#        'Y_te', 'location_group', 'features', 'location_ids'],
#       dtype='object')

# TODO 1: What is this location_group?
print(loaded_data['location_group'])

# TODO 2: How to understand this features??
print(loaded_data['features'])

# TODO 3: How to handle this location_ids?? Cluster? Segmentation?
print(loaded_data['location_ids'])





X_tr = loaded_data['X_tr']
Y_tr = loaded_data['Y_tr']
X_te = loaded_data['X_te']
Y_te = loaded_data['Y_te']

location_group = loaded_data['location_group']
features = loaded_data['features']
location_ids = loaded_data['location_ids']


