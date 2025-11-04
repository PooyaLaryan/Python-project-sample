import numpy as np
from scipy import stats
import random

data = random.sample(range(1, 30), 10)
zscore = stats.zscore(data)

print(zscore)