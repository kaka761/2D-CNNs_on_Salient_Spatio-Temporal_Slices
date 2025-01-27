import os
import numpy as np
import random

idx = [i for i in range(15)]
random.shuffle(idx)
#val = random.sample(idx, 8)
print(idx)
train = idx[0:9]
val = idx[9:12]
test = idx[12:]
print('train =', train)
print('val =', val)
print('test =', test)