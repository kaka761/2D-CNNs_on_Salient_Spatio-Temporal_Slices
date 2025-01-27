import cv2
import numpy as np
import csv
import pandas as pd
import os
import glob
from matplotlib import pyplot as plt


def count_f(filename):
    cap = cv2.VideoCapture(filename)
    d = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    i = 0
    for fn in range(d):
        ret, frame = cap.read()
        if ret is False:
            continue
        i +=1
    cap.release()
    return i


def get_xy(filename, new_path):
    cap = cv2.VideoCapture(filename)
    d = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    i=0
    c_df = 0
    print('ori', d, w, h)
    while(cap.isOpened()):
        ret, frame = cap.read() # read a video frame
        if not ret: break
        #frame = cv2.resize(frame, (320, 240))
        file_path = new_path + '/' + filename.split('/')[-1].split('.')[0]
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        print(file_path)
        i += 1
        cv2.imwrite(file_path + '/' + filename.split('/')[-1].replace('.avi', '_') + str("{:0>6d}".format(i)) + ".jpg", frame)
    cap.release()
    return d


def get_yt(filename, new_path):#, n_frames= 1):
    start_slicing = False
    cap = cv2.VideoCapture(filename)
    d = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print('ori', d, w, h)
    #w = 320
    #h = 240
    t, y = np.meshgrid(np.arange(d), np.arange(h)) # for yt
    list = []
    i = 0    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret: 
            break
        #frame = cv2.resize(frame, (320, 240))
        print('i', i)
        file_path = new_path + '/' + filename.split('/')[-1].split('.')[0]
        if not os.path.exists(file_path):
            os.mkdir(file_path)   
        print(file_path)     
        i += 1
        list.append(frame)
        print('len', len(list))
        if not start_slicing and i%(d) == 0:
            print("Dimension full. Begin slicing.")
            start_slicing = True
        if start_slicing:
            a = 0
            for j in range(w):
                a += 1
                #print(a)
                l = np.array(list).astype(np.uint8)
                slice = l[t, y, j]
                cv2.imwrite(file_path + '/' + filename.split('/')[-1].replace('.avi', '_') + str("{:0>6d}".format(a)) + ".jpg", slice)
    cap.release()
    return w # for xt


def get_xt(filename, new_path):#, n_frames= 1):
    start_slicing = False
    cap = cv2.VideoCapture(filename)
    d = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print('ori', d, w, h)
    x, t= np.meshgrid(np.arange(w), np.arange(d)) # for xt
    i = 0
    list = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret: # or i == r_d: 
            break
        #frame = cv2.resize(frame, (320, 240))
        print('i', i)
        file_path = new_path + '/' + filename.split('/')[-1].split('.')[0]
        if not os.path.exists(file_path):
            os.mkdir(file_path)   
        print(file_path)     
        i += 1
        list.append(frame)
        # if not start_slicing and i%(r_d) == 0:
        print('len', len(list))
        if not start_slicing and i%(d) == 0:
            print("Dimension full. Begin slicing.")
            start_slicing = True
        if start_slicing:
            a = 0
            for j in range(h):
                a += 1
                l = np.array(list).astype(np.uint8)
                slice = l[t, j, x]
                cv2.imwrite(file_path + '/' + filename.split('/')[-1].replace('.avi', '_') + str("{:0>6d}".format(a)) + ".jpg", slice)
    cap.release()
    print(i)
    return h # for xt


def store_frames(frames, path2store, name):
    print('path2store:', path2store+name)
    path = path2store + name
    if not os.path.exists(path):
        os.mkdir(path)
    for ii, frame in enumerate(frames):
        print(ii)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
        path2img = os.path.join(path, name + '_' + str("{:0>3d}".format(ii))+".jpg")
        cv2.imwrite(path2img, frame)



save_path = './Datasets/AR/hmdb51img/' 
path = './Datasets/AR/hmdb51/' 
#files = sorted(glob.glob(path))

### get_slices ###
for root, dirs, files in os.walk(path):
        
    if len(root.split('/')) > 6:
        for f in files:
            v_path = root + '/' + f
            print(v_path)
            
            new_path = save_path + v_path.split('/')[-2]
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            print(new_path)
            #T = get_xt(v_path, new_path)
            #T = get_yt(v_path, new_path)
            T = get_xy(v_path, new_path)
