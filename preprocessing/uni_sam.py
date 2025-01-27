import os
import shutil#
import numpy as np
import random


def get_seq_frames(video_size, num_frames, start_index=0, max_frame=-1): ### start_index=1, max_frame=-1
    seg_size = max(0., float(video_size - 1) / num_frames) # 140/4 = 35
    if max_frame == -1:
        max_frame = int(video_size) - 1 # 140
    seq = []
    for i in range(num_frames):
        start = int(np.round(seg_size * i))     # 0  35  70  105
        end = int(np.round(seg_size * (i + 1)))  # 35 70  105 140
        idx = min(random.randint(start, end) + start_index, max_frame-1) ###  max_frame-1
        seq.append(idx)
    return seq

path = './Datasets/AR/hmdb51img/'
new_path = './Datasets/AR/hmdbimg24/'
for root, dirs, files in os.walk(path):
    if len(root.split('/')) > 7:
        l = len(files)#//4
        print('len_video', l)
        #frame_list = get_seq_frames(l, 32, start_index=0, max_frame=-1)
        frame_list = np.linspace(0, l-1, 24, dtype=np.int16)
        for i in range(l):
            if (files[i][-3:] == 'jpg') or (files[i][-3:] == 'png') or (files[i][-3:] == 'JPG'):
                if i in frame_list:
                    file_path = root+'/'+files[i]
                    print(file_path)
                    new_file_path = new_path + root.split('/')[6] + '/' + root.split('/')[7]
                    if not os.path.exists(new_file_path):
                        os.makedirs(new_file_path)
                    new_file = new_file_path + '/' + files[i]
                    shutil.copy(file_path, new_file_path)
print('done')