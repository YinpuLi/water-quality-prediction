import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)




from utils.utils import get_absolute_path, load_data, save_csv, data_prep_X, gen_col_name, data_pre_Y
from utils.constants import *

data_path = get_absolute_path('water_dataset.mat', 'data')
data = load_data(data_path)

x_train = data['X_tr']


print(type(data))
print(data.keys())


x_test = data['X_te']
y_train= data['Y_tr']
y_test= data['Y_te']


# feature_sets = data['features']
feature_cols = gen_col_name(p=11)


df_train = data_prep_X(x_train)
df_test = data_prep_X(x_test)
df_train.columns = df_test.columns = feature_cols


y_train = data_pre_Y(y_train)
y_test = data_pre_Y(y_test) 
y_train.columns = y_test.columns = ['measurement']



print(df_train.head())
print(y_test.head()) # (15651, 11): 423 * 37

print(df_test.shape) # (10434, 11)
print( y_test.shape)


save_csv(df_train, 'data/X_train.csv')
save_csv(y_train, 'data/y_train.csv')
save_csv(df_test, 'data/X_test.csv')
save_csv(y_test,  'data/y_test.csv')

