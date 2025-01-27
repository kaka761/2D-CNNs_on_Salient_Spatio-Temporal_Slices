import os
import pandas as pd
import shutil

def store_xy(save_path, df, c_df):#, n_frames= 1):
    root = df[c_df][0]
    d = root.split('/')[-1].split('.')[0]
    start = int(df[c_df][-3])
    i = start
    name = df[c_df][1]
    n = df[c_df][-1]
    file_path = save_path + name
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    final_path = file_path + '/' + d + '_' + str(start)
    if not os.path.exists(final_path):
        os.mkdir(final_path)        
    for a in range(n):
        img_name = d + '_' + str("{:0>6d}".format(i)) + '.jpg'
        p = os.path.join(root, img_name)
        i += 1
        print('p', p)
        new_path = final_path + '/' + img_name
        print(new_path)
        shutil.copy(p, new_path)
    return i
        

# get a name for the video
save_path = './Datasets/IPN/slices/xy/trainf/' #yt_feet_264/'
path_csv = './Projects/slice_ar/csv/ipn/trainf.csv'

df = pd.read_csv(path_csv)
df1 = df.values
### get_slices ###
for c_df in range(len(df1)):
    i = store_xy(save_path, df1, c_df)
