import pickle
import torch
import numpy as np
import os
import csv
from collections import Counter
import pandas as pd


def NUM(path):
    num_f = []
    with open(path, 'r', newline = '') as f:
        reader = csv.reader(f)
        i = 0
        for r in reader:
            num_f.append(int(r[-1]))      
    i = sum(num_f)
    return (i, num_f)



srate = 1
path = './Projects/slice_ar/csv/nhg/oo_vaNum.csv'
f = open(r'./Projects/slice_ar/results/nhg/oo/28.pkl', 'rb')
bs = 256
data = pickle.load(f)
seq = 4

sumf, num_xt = NUM(path) #num(path)

for k, v in data.items():
    print('key', k)
    if k == 'img_name':
        names = v
    if k == 'preds':
        preds = v
    if k == 'labels':
        labels = v

print('img_names', len(names)) # 356x128
print('preds', len(preds))
print('labels', len(labels))

N = []
P = np.zeros(sumf)
L = np.zeros(sumf)
Pr = []
La = []

for i in range(len(names)-1):
    N += names[i]
    P[i*bs:(i+1)*bs] = preds[i]
    L[i*bs:(i+1)*bs] = labels[i]
P[(i+1)*bs:] = preds[i+1]
L[(i+1)*bs:] = labels[i+1]

cors = 0
for j in range(len(P)):
    if P[j] == L[j]:
        cors += 1
print('frame_acc,',cors/len(P)) 

corrects = 0
m = 0
#a = 0
for i in num_xt:
    valp, countp = np.unique(P[m:m+i], return_counts = True)
    vall, countl = np.unique(L[m:m+i], return_counts = True)
    m = m+i
    idxp = np.argmax(countp)
    idxl = np.argmax(countl)
    if valp[idxp] == vall[idxl]:
        corrects += 1      
    Pr.append(valp[idxp])
    La.append(vall[idxl])
print('videos_acc,',corrects/len(Pr))