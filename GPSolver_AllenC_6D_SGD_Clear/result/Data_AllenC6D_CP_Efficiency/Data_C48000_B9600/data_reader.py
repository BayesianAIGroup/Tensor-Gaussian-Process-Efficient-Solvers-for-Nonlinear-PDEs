import numpy as np


data = np.load("process2.npz")

for k in data:
    print(k)
    print(data[k])