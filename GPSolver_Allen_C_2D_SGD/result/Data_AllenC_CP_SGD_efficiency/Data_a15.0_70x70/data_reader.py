import numpy as np


data = np.load("process1.npz")

for k in data:
    print(k)
    print(data[k])