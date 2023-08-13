import pandas as pd
import os
import sys
import ast

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.utils import load_data


data = load_data('data/water_dataset.mat')

# Convert string representation of arrays to actual arrays
for key in data:
    if key.endswith('_tr') or key.endswith('_te'):
        data[key] = ast.literal_eval(data[key])

# Create a DataFrame
df = pd.DataFrame(data)

# Save DataFrame to a CSV file
df.to_csv('output.csv', index=False)


# print()

# # TODO 1: What is this location_group?
# print(loaded_df['location_group'])

# # TODO 2: How to understand this features??
# print(loaded_df['features'])

# # TODO 3: How to handle this location_ids?? Cluster? Segmentation?
# print(loaded_df['location_ids'])

# location_group = loaded_df['location_group']
# features = loaded_df['features']
# location_ids = loaded_df['location_ids']



# X_tr = pd.DataFrame(loaded_df['X_tr'])
# Y_tr = pd.DataFrame(loaded_df['Y_tr'])
# X_te = pd.DataFrame(loaded_df['X_te'])
# Y_te = pd.DataFrame(loaded_df['Y_te'])

# print(X_tr.head())
# print(X_tr.columns)

# X_tr.to_csv('data/X_tr.csv')

