from numpy import random
import numpy as np
from matplotlib import pyplot as plt
# print(np.arange(2,15))
# print(random.choice((np.arange(2,14))))
h = [154, 270, 370.5, 438.5, 480, 517.5, 552.5, 580.5, 585, 570, 589.5, 561, 567.5, 525, 525, 502.5, 459, 448, 442, 360]
x = np.arange(1,21)
plt.plot(x,h)
plt.xlabel("Number of harvests")
plt.ylabel("Total reward over the episode")
plt.title("Total rewards vs number of harvests(leave time = 10 sec)")
plt.savefig("lt10.jpg")
plt.close()

h2 = [420, 648, 780, 863.5, 900, 931.5, 924, 924, 900, 889.5, 855, 816, 780, 748.5, 717, 675, 612, 589.5, 522.5, 497.5]
plt.plot(x,h2)
plt.xlabel("Number of harvests")
plt.ylabel("Total reward over the episode")
plt.title("Total rewards vs number of harvests(leave time = 3 sec)")
plt.savefig("lt3.jpg")
# Q = [1,1,1,2,2,2,2,3,3,3,4,4,4,4,4]
# max_indices=np.where(Q==np.amax(Q))
# print(max_indices[0])