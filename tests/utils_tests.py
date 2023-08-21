import os
import sys


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)



from utils.utils import *
from utils.constants import *



data_path = get_absolute_path('water_dataset.mat', 'data')
loaded_data = load_data(data_path)

updated_loaded_data = add_temporal_spatial_cols(loaded_data)



print(updated_loaded_data.keys())

temp_df = pd.DataFrame(updated_loaded_data['X_tr'][0][len(updated_loaded_data['X_tr'][0])-1])
print(updated_loaded_data['X_tr'][0][0].shape)
print(temp_df.shape)
print(temp_df.columns)
print(temp_df)

print()
print()
print()

temp_df = pd.DataFrame(updated_loaded_data['X_te'][0][len(updated_loaded_data['X_te'][0])-1])
print(updated_loaded_data['X_te'][0][1].shape)
print(temp_df.shape)
print(temp_df.columns)
print(temp_df)

# # print(loaded_data['X_tr'][0][0].shape)

# # print(loaded_data['X_tr'][0][0].shape)

# # print(type(loaded_data['location_ids']))




# # print(get_train_dates_col(datetime.strptime(train_start, "%Y-%m-%d"), 1))# ['2016-01-28']

# # print(get_test_dates_col(datetime.strptime(test_end, "%Y-%m-%d"), num_days=1)) # ['2018-01-01']



# # print(len(get_train_dates_col()))# 432
# # print(len(get_test_dates_col()))# 282