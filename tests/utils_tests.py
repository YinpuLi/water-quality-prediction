import os
import sys


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)



from utils.utils import *
from utils.constants import *





print(get_train_dates_col(datetime.strptime(train_start, "%Y-%m-%d"), 1))# ['2016-01-28']

print(get_test_dates_col(datetime.strptime(test_end, "%Y-%m-%d"), num_days=1)) # ['2018-01-01']



print(len(get_train_dates_col()))# 432
print(len(get_test_dates_col()))# 282