import os
import sys


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)



from utils.utils import *
from utils.constants import *



data_path = get_absolute_path('water_dataset.mat', 'data')
loaded_data = load_data(data_path)


updated_loaded_data = add_temporal_spatial_cols(loaded_data)
stacked_X_tr, stacked_X_te = stack_dataframes(updated_loaded_data)


print(stacked_X_tr.head)

# Set the column names of the stacked_X_tr, stacked_X_te dataframes
stacked_X_tr.columns = column_names_extended 
stacked_X_te.columns = column_names_extended


print(stacked_X_tr.columns)
print(stacked_X_tr.head)

stacked_X_tr = stacked_X_tr.astype(column_data_extended_types)
stacked_X_te = stacked_X_te.astype(column_data_extended_types)


print(stacked_X_tr.head)
print(stacked_X_te.head)


print(stacked_X_tr.shape)
print(stacked_X_te.shape)
# (15651, 16)
# (10434, 16)


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