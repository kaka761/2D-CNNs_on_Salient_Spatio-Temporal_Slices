import cv2
import numpy as np
import os
import csv


csv_path = './Projects/slice_ar/csv/kth/coor.csv' 
path = './Datasets/KTH/videos' 

for root, dirs, files in os.walk(path):
    if len(root.split('/')) > 6:
        for f in files:
            data = []
            v_path = root + '/' + f
            print(v_path)
            data.append(v_path)
            cap = cv2.VideoCapture(v_path)
            h= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            data.append(0)
            data.append(0)
            data.append(w)
            data.append(h)
            with open(csv_path, 'a', newline='') as fw:
                w = csv.writer(fw)
                w.writerow(data)