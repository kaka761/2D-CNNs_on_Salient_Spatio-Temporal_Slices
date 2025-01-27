from genericpath import exists
from xmlrpc.client import _iso8601_format
import cv2
import numpy as np
from skimage.morphology import dilation
import os
import shutil

class JBaseSaliencyCrops():
    def __init__(self, crop_size, num_crops, **kwargs):
        self.n = num_crops
        self.cs = crop_size

        # 3, 5, 15, 0

        self.pre_sigma = kwargs.get('pre_sigma', 3.)
        self.post_sigma = kwargs.get('post_sigma', 5.)
        self.k = kwargs.get('k', 15)
        self.alpha = kwargs.get('alpha', 0.)

    def _compute_J(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.GaussianBlur(image, (0, 0), self.pre_sigma) / 255.

        # same as in tutorial
        dx = cv2.Sobel(image, -1, 1, 0) # x direction
        dy = cv2.Sobel(image, -1, 0, 1) # y direction

        Ixx = dx**2.
        Ixy = dx * dy
        Iyy = dy**2.
        
        Ixx = cv2.GaussianBlur(Ixx, (0, 0), self.post_sigma)
        Ixy = cv2.GaussianBlur(Ixy, (0, 0), self.post_sigma)
        Iyy = cv2.GaussianBlur(Iyy, (0, 0), self.post_sigma)
        
        det = Ixx * Iyy - Ixy**2
        trace = Ixx + Iyy

        return det - self.alpha * trace

    def __call__(self, image):

        if not isinstance(image, type(np.array)):
            image = np.array(image)

        struct_tensor = self._compute_J(image)
        Y = find_local_max(struct_tensor, k=self.k)

        # returns: listx pos / y pos
        top_k = np.unravel_index(
            np.argsort(-Y.flatten())[:self.n], Y.shape
        )

        patches_ = []
        for i in range(top_k[0].shape[0]):
            y, x = top_k[0][i], top_k[1][i]
            # from middle to top left
            x -= self.cs // 2
            y -= self.cs // 2

            x = max(x, 0)
            x = min(x, image.shape[1] - self.cs)

            y = max(y, 0)
            y = min(y, image.shape[0] - self.cs)

            patch = image[y:y + self.cs, x:x + self.cs, :]
            #print(patch)
            patches_.append(patch)

        return patches_

def find_local_max(keypoints, k=31):
    kernel = np.ones((k, k))
    # set value in the middle zero
    kernel[k // 2, k // 2] = 0
    # for a pixel in the middle (i,j) subtract the max-value in the nhood
    local_max = (keypoints - dilation(keypoints, kernel)).clip(min=0)

    return local_max

def store_frames(frames, path2store, name):
    for ii, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
        path2img = os.path.join(path2store, name.split('.')[0] + '_' + str("{:0>3d}".format(ii))+".jpg")
        cv2.imwrite(path2img, frame)


root = './Datasets/NHG/slices/xy/'
target = './Datasets/NHG/slices/xys16/'
idx = []
for root, dirs, files in os.walk(root, topdown=False):
    if len(root.split('/')) > 8:
        print(root)
        dic = {}
        for name in files:
            path = os.path.join(root, name)
            img = cv2.imread(path)
            saliency = JBaseSaliencyCrops(crop_size=112, num_crops=3, image=img)
            struct_tensor = saliency._compute_J(img)
            a = np.sum(struct_tensor)
            dic[path] = a
        dic_sorted = sorted(dic.items(), key = lambda x:x[1], reverse = True)
        for i in dic_sorted:
            out = i[1]
            #if i[1]<0.1:
                #idx.append(int(i[0].split('.')[0].split('_')[-1]))
                #print(idx)
                #break
        out = dic_sorted[0:16]
        for i in out:
            out1 = i[0]
            print(out1)
            
            save_path = target + str(out1.split('/')[7]) 
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            new_path = save_path + '/' + str(out1.split('/')[8])
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            shutil.copy(out1,new_path)