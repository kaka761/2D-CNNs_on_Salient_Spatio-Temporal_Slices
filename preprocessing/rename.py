import os
import csv

     
img_path = './Datasets/AR/hmdbimg24/'
for root, dirs, files in os.walk(img_path):
    if len(root.split('/')) > 7:
        print(root)
        l = len(files)
        f = os.listdir(root)
        f.sort()
        #print(f)
        i = 0
        for i in range(l):
            src = root + '/' + f[i]
            dst = root + '/' + str("{:0>2d}".format(i)) + '.jpg'
            i += 1
            try:
                os.rename(src, dst)
                print('rename from %s to %s', (src, dst))
            except:
                continue
    print('ending')
            