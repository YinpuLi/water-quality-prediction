
p = 37 # num of features
K = 37 # spacial; num of locations

n_tr = 423 # num of contiguous dates in training data
n_te = 282 # num of contiguous dates in testing data



train_start = "2016-01-28"
test_end = '2018-01-01'

column_data_extended_types = {
    'Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius (Maximum)': float,
    'pH, water, unfiltered, field, standard units (Maximum)': float,
    'pH, water, unfiltered, field, standard units (Minimum)': float,
    'Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius (Minimum)': float,
    'Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius (Mean)': float,
    'Dissolved oxygen, water, unfiltered, milligrams per liter (Maximum)': float,
    'Dissolved oxygen, water, unfiltered, milligrams per liter (Mean)': float,
    'Dissolved oxygen, water, unfiltered, milligrams per liter (Minimum)': float,
    'Temperature, water, degrees Celsius (Mean)': float,
    'Temperature, water, degrees Celsius (Minimum)': float,
    'Temperature, water, degrees Celsius (Maximum)': float,
    'Date': str,
    'Location_ID': str,
    # 'Year': str,
    'Month': str,
    'Week': str,
    'Weekday': str,
    'Season': str,
    'Season_Num': int
}


column_names_raw = [
    'Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius (Maximum)',
    'pH, water, unfiltered, field, standard units (Maximum)',
    'pH, water, unfiltered, field, standard units (Minimum)',
    'Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius (Minimum)',
    'Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius (Mean)',
    'Dissolved oxygen, water, unfiltered, milligrams per liter (Maximum)',
    'Dissolved oxygen, water, unfiltered, milligrams per liter (Mean)',
    'Dissolved oxygen, water, unfiltered, milligrams per liter (Minimum)',
    'Temperature, water, degrees Celsius (Mean)',
    'Temperature, water, degrees Celsius (Minimum)',
    'Temperature, water, degrees Celsius (Maximum)'
]

column_names_extended = [
    'Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius (Maximum)',
    'pH, water, unfiltered, field, standard units (Maximum)',
    'pH, water, unfiltered, field, standard units (Minimum)',
    'Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius (Minimum)',
    'Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius (Mean)',
    'Dissolved oxygen, water, unfiltered, milligrams per liter (Maximum)',
    'Dissolved oxygen, water, unfiltered, milligrams per liter (Mean)',
    'Dissolved oxygen, water, unfiltered, milligrams per liter (Minimum)',
    'Temperature, water, degrees Celsius (Mean)',
    'Temperature, water, degrees Celsius (Minimum)',
    'Temperature, water, degrees Celsius (Maximum)',
    'Date',
    'Location_ID',
    # 'Year',
    'Month',
    'Week',
    'Weekday',
    'Season',
    'Season_Num'
]


EVAL_METRIC_LIST = ['rmse','mape','wmape','wbias','wuforec','woforec']

RANDOM_SEED = 827 # global random_state