import numpy as np

a = np.zeros((12,4,12))

for i in range(12):
    for j in range(4):
        for k in range(12):
            a[i,j,k] = 5
            
print(np.sum(a[5][2]))