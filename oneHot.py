import random
import pandas as pd
import numpy as np


data_list = np.array([1,1,1,3,6,7,1,6,7,3,3,7])
data_dummies = pd.get_dummies(data_list)
print(data_dummies)