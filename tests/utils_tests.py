import os
import sys


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)



from utils.utils import *
from utils.constants import *



data_path = get_absolute_path('water_dataset.mat', 'data')
loaded_data = load_data(data_path)
# Extract the feature names from loaded_data['features']
feature_names = [name[0] for name in loaded_data['features'][0]]

updated_loaded_data = add_temporal_spatial_cols(loaded_data)
stacked_X_tr, stacked_X_te = stack_dataframes(updated_loaded_data)
# Set the column names of the stacked_X_tr, stacked_X_te dataframes
stacked_X_tr.columns = feature_names + ['Date', 'Location_ID']
stacked_X_te.columns = feature_names + ['Date', 'Location_ID']



print(stacked_X_tr.head)
print(stacked_X_te.head)

print(stacked_X_te.columns)