import random
import pandas as pd
import numpy as np
data = [random.randint(1,10) for i in range(10)]
print(data)
pdata = pd.DataFrame(data)
mean = pdata.mean().values
print('mean :',mean)
s = data - mean
print('s :',s)
s2 = s ** 2
print('s2 :',s2)
sum = s2.sum()
print('sum :',sum)
v1 = sum / (len(data) -1) 
print('v1 :',v1)

v2 = sum / (len(data)) 
print('v2 :',v2)

v = np.var(data)
print('v :',v)

