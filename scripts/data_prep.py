import os
import sys


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)



from utils.utils import *
from utils.constants import *



data_path = get_absolute_path('water_dataset.mat', 'data')
loaded_data = load_data(data_path)
# Extract the feature names from loaded_data['features']
feature_names = [name[0] for name in loaded_data['features'][0]] + ['Date', 'Location_ID']
updated_loaded_data = add_temporal_spatial_cols(loaded_data)
stacked_X_tr, stacked_X_te = stack_dataframes(updated_loaded_data)
# Set the column names of the stacked_X_tr, stacked_X_te dataframes
stacked_X_tr.columns = feature_names 
stacked_X_te.columns = feature_names

stacked_X_tr = stacked_X_tr.astype(column_data_types)
stacked_X_te = stacked_X_te.astype(column_data_types)


print(stacked_X_tr.head)
print(stacked_X_te.head)


print(stacked_X_tr.shape)
print(stacked_X_te.shape)
# (15651, 13)
# (10434, 13)


stacked_X_tr_path = get_absolute_path(
    'stacked_X_tr.csv'
    , rel_path='data'
)

stacked_X_te_path = get_absolute_path(
    'stacked_X_te.csv'
    , rel_path='data'
)
save_csv(stacked_X_tr, stacked_X_tr_path)
save_csv(stacked_X_te, stacked_X_te_path)