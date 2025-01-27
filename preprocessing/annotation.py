import os
import shutil
import csv
import pandas as pd
from sklearn.utils import shuffle

'''
### CAM ###
train = [16, 10, 11, 0, 3, 6, 7, 8, 17, 9, 4, 12]
val = [15, 19, 1, 13]
test = [14, 2, 18, 5]
'''
train = [0, 11, 13, 5, 12, 7, 9, 3, 6]
val = [10, 2, 8]
test = [4, 1, 14]


img_path = './Datasets/Northwestern_Hand_Gesture/slices/xyt'
csv_path = './Projects/slice_ar/csv/nhg/oo_train.csv'
for root, dirs, files in os.walk(img_path):
    if len(root.split('/'))> 8:
        if int(root.split('_')[4]) in train:
            print(root)
            fs = os.listdir(root)
            fs.sort()
            length = len(fs)
            # length = (len(fs) // 4)*4
            for i in range(length):
                data = []
                img = root + '/' + fs[i]
                data.append(img)
                label = root.split('/')[7]
                data.append(label)
                ### for xy_xt ###
                #xt = img.split('xy')[0] + 'xt' + img.split('xy')[1]
                #data.append(xt)
                #yt = img.split('xy')[0] + 'yt' + img.split('xy')[1]
                #data.append(yt)
                ###-----------###
                with open(csv_path, 'a', newline = '') as f:
                    w = csv.writer(f)
                    w.writerow(data)
                    