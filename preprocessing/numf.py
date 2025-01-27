import os
import numpy as np


path = './Datasets/IPN/slices/xy/' 
#files = sorted(glob.glob(path))
num_f = []
### get_slices ###
for root, dirs, files in os.walk(path):
    
    if len(root.split('/')) > 9:
        print(root)
        for f in files:
            v_path = root + '/' + f
        num_f.append(len(files))
print(np.max(np.array(num_f)))
print(np.min(np.array(num_f)))
print(np.mean(np.array(num_f)))