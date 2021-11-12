import numpy as np
from numpy import random

random.seed(51)
P = random.choice([random.randint(2,7), random.randint(10,15)], size = 10)
print(P)