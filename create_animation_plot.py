import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("GDPs.csv") 
fig=plt.figure(dpi=200)
plt.grid()
plt.xlabel('Year')
plt.ylabel('GDP (in $ Trillion)')
plt.xlim(1968,2022) 
plt.ylim(-1,25)

for i in range(1971,2021,1):
    dfi=df[df["year"]<=i]  
    plt.plot(dfi["year"],dfi["USA"],color='b',linestyle='-') 
    plt.plot(dfi["year"],dfi["China"],color='r',linestyle='--')
    plt.plot(dfi["year"],dfi["Japan"],color='m',linestyle='-.')
    plt.plot(dfi["year"],dfi["Germany"],color='k',linestyle=':')
    plt.plot(dfi["year"],dfi["UK"],color='g',linestyle=None)
    plt.legend(labels=["USA","China","Japan","Germany","UK"],
               loc="upper left")
    plt.title(f'GDPs till Year {i}') 
    plt.savefig(f"files/ch01/year{i}.png")



import PIL
import imageio
import numpy as np

frames=[] 
for i in range(1971,2021,1):
    frame=PIL.Image.open(f"files/ch01/year{i}.png")  
    frame=np.asarray(frame)
    frames.append(frame) 
imageio.mimsave('files/ch01/GDPs.gif', frames, fps=5)