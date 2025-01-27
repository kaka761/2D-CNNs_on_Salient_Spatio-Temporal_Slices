import os
import shutil

img_path = './Datasets/Northwestern_Hand_Gesture/videos1'
save_path = './Datasets/Northwestern_Hand_Gesture/videos'

for root, dirs, files in os.walk(img_path):
    if len(root.split('/')) > 8:
        print(root)
        l = len(files)
        for f in files:
            if len(f.split('.')) == 2:
                src = os.path.join(root,f)
                new_path = save_path + '/' + root.split('/')[7]
                if not os.path.exists(new_path):
                    os.mkdir(new_path)
                new_path1 = new_path + '/' + root.split('/')[8]
                if not os.path.exists(new_path1):
                    os.mkdir(new_path1)
                final_path = new_path1 + '/' +f
                print(final_path)
                shutil.copy(src, final_path)