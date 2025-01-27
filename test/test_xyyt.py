import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, SubsetRandomSampler
from torch.nn import DataParallel
import os
from PIL import Image, ImageOps
import time
import pickle
import numpy as np
from torchvision.transforms import Lambda
import argparse
import copy
import random
import numbers
import math
import cv2
from torch.utils.data import Sampler
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import glob
import pandas as pd

parser = argparse.ArgumentParser(description='resnet training')
parser.add_argument('-g', '--gpu', default=[0], nargs='+', type=int, help='index of gpu to use, default 2')
parser.add_argument('-s', '--seq', default=16, type=int, help='sequence length, default 4')
parser.add_argument('-t', '--test', default=256, type=int, help='train batch size, default 100')
parser.add_argument('-o', '--opt', default=1, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-w', '--work', default=4, type=int, help='num of workers to use, default 2')
parser.add_argument('-f', '--flip', default=0, type=int, help='0 for not flip, 1 for flip, default 0')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
parser.add_argument('--weightdecay', default=0, type=float, help='weight decay for sgd, default 0')
parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd, default 0')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum, default False')
parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
parser.add_argument('--sgdstep', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
parser.add_argument('--sgdgamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')
parser.add_argument('--srate', default=1, type=int, help='sample rate')

parser.add_argument('--anno_t', default='./Projects/slice_ar/csv/cam/xyxt_test.csv', type=str, help='videos path')
parser.add_argument('--save_path', default='./Projects/slice_ar/results/cam/xyxt/', type=str, help='results path')
parser.add_argument('--test_path', default='./Projects/slice_ar/results/cam/xyxt/best.pth', type=str, help='test path')

args = parser.parse_args()

gpu_usg = ",".join(list(map(str, args.gpu)))
sequence_length = args.seq
test_batch_size = args.test
optimizer_choice = args.opt
multi_optim = args.multi
workers = args.work
use_flip = args.flip
crop_type = args.crop
momentum = args.momentum
weight_decay = args.weightdecay
dampening = args.dampening
use_nesterov = args.nesterov

sgd_adjust_lr = args.sgdadjust
sgd_step = args.sgdstep
sgd_gamma = args.sgdgamma

anno_t = args.anno_t
srate = args.srate

save_path = args.save_path
test_path = args.test_path

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()

print('number of gpu   : {:6d}'.format(num_gpu))
print('test batch size: {:6d}'.format(test_batch_size))
print('test crop type  : {:6d}'.format(crop_type))
print('whether to flip : {:6d}'.format(use_flip))
def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            #img = img.crop((0, 0, img.size[0] / 2, img.size[1]))
            return img.convert('RGB')


class RandomCrop(object):

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.count = 0

    def __call__(self, img):

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        # print(self.count, x1, y1)
        self.count += 1
        return img.crop((x1, y1, x1 + tw, y1 + th))


class Dataset(Dataset):
    def __init__(self, file_paths, file_paths2, file_labels,
                 transform=None, loader=pil_loader):# kdata,
        self.file_paths = file_paths
        self.file_paths2 = file_paths2
        self.file_labels_phase = file_labels
        self.transform = transform
        self.loader = loader
        self.frames = []

    def __getitem__(self, index):
        img_names = self.file_paths[index] # '/home/yaxin/test/weizmann/xy/daria_bend_010.jpg'
        img_names2 = self.file_paths2[index]
        labels_phase = self.file_labels_phase[index]
        #data = self.kdata[index]
        #print(img_names)
        #print(labels_phase)
        imgs = self.loader(img_names)
        imgs2 = self.loader(img_names2)
        if self.transform is not None:
            imgs = self.transform(imgs)
            imgs2 = self.transform(imgs2)
        return imgs, imgs2, labels_phase, img_names.split('/')[-1]#, data''''''
    def __len__(self):
        return len(self.file_paths)


class resnet_lstm(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm, self).__init__()  # 继承ResNet网络结构
        resnet = models.resnet18(pretrained=False)  # from torchvision import models
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1) # 64
        self.share.add_module("layer2", resnet.layer2) # 128
        self.share.add_module("layer3", resnet.layer3) # 256
        self.share.add_module("layer4", resnet.layer4) # 512
        self.share.add_module("avgpool", resnet.avgpool)
        #self.lstm = nn.LSTM(512, 128, batch_first=True)
        self.fc = nn.Linear(512*2, 10)

        #init.xavier_normal_(self.lstm.all_weights[0][0])  # 没有预训练，则使用xavier初始化
        #init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x, x1): 
        x = self.share.forward(x)
        x = x.view(-1, 512)
        x1 = self.share.forward(x1)
        x1 = x1.view(-1, 512) # torch.Size([256, 512])
        y = torch.cat([x, x1], dim = 1)
        #x = x.view(-1, 26, 512)
        #self.lstm.flatten_parameters()
        #y, _ = self.lstm(x)
        #y = y.contiguous().view(-1, 128)
        y = self.fc(y) 
        return y




def get_data():
    df_test= pd.read_csv(anno_t, header = None)
    dft = df_test.values
    test_paths = dft[:, 0]
    test_labels = dft[:, 1]
    test_paths2 = dft[:, 2]


    print('valid_paths  : {:6d}'.format(len(test_paths)))
    print('valid_labels : {:6d}'.format(len(test_labels)))
    '''
    ### for IPN ###
    label_dict = {'D0X':0, 'B0A':1, 'B0B':2, 'G01':3, 'G02':4, 'G03':5, 'G04':6, 'G05':7, 'G06':8, 
                  'G07':9, 'G08':10, 'G09':11, 'G10':12, 'G11':13}
    
    label_dict = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8}   
    '''
    label_dict = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9}
    test_labels = [label_dict[l] for l in test_labels]
       

    test_labels = np.asarray(test_labels, dtype=np.int64)
    ### ------- ### 

    test_transforms = transforms.Compose([
        #transforms.Resize([112, 112]),
        transforms.Resize([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
    ])

    test_dataset = Dataset(test_paths, test_paths2, test_labels, test_transforms)

    return test_dataset


def test_model(test_dataset):
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        #sampler=test_idx,
        num_workers=workers,
        pin_memory=False
    )
    model = resnet_lstm()
    model = DataParallel(model)
    model.load_state_dict(torch.load(test_path))

    if use_gpu:
        model = model.cuda()
    # 应该可以直接多gpu计算
    # model = model.module            #要测试一下
    #criterion = nn.CrossEntropyLoss(size_average=False)

    model.eval()
    test_corrects = 0
    test_start_time = time.time()

    all_preds = []
    test_num = 0
    val_name = []
    val_pred = []
    val_label = []
    for data in test_loader:
        inputs, inputs1, labels_phase, img_names  = data
        if use_gpu:
            inputs = Variable(inputs.cuda())
            inputs1 = Variable(inputs1.cuda())
            labels = Variable(labels_phase.cuda())
            #kdatas = Variable(kdata.cuda())
            img_names = img_names 
        else:
            inputs = Variable(inputs)
            inputs1 = Variable(inputs1.cuda())
            labels = Variable(labels_phase)
            #kdatas = Variable(kdata)

        if crop_type == 0 or crop_type == 1:
            #outputs = model.forward(inputs, kdatas)
             outputs = model.forward(inputs, inputs1)
        elif crop_type == 5:
            inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
            inputs = inputs.view(-1, 3, 224, 224)
            #outputs = model.forward(inputs, kdatas)
            outputs = model.forward(inputs)
            outputs = outputs.view(5, -1, 3)
            outputs = torch.mean(outputs, 0)
        elif crop_type == 10:
            inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
            inputs = inputs.view(-1, 3, 224, 224)
            #outputs = model.forward(inputs, kdatas)
            outputs = model.forward(inputs, inputs1)
            outputs = outputs.view(10, -1, 3)
            outputs = torch.mean(outputs, 0)

        outputs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs.data, 1)
        for i in range(len(preds)):
            all_preds.append(preds[i])
        print(len(all_preds))
        #loss = criterion(outputs, labels)
        #test_loss += loss.data[0]
        ###
        val_name.append(img_names)
        val_pred.append(preds.cpu().numpy())
        val_label.append(labels.cpu().numpy())
        ###
        test_corrects += torch.sum(preds == labels.data)
        test_num += labels.shape[0]
    test_elapsed_time = time.time() - test_start_time
    test_accuracy = test_corrects / test_num
   # test_average_loss = test_loss / test_num

    print('type of all_preds:', type(all_preds))
    print('leng of all preds:', len(all_preds))
    save_test = int("{:4.0f}".format(test_accuracy * 10000))
    pred_name = save_path + 'test_' + str(save_test) + '.pkl'
    with open(pred_name, 'wb') as f:
        pickle.dump(all_preds, f)    

    ### save .pkl ###
    file_name = save_path + 'test.pkl'
    with open(file_name,'wb') as f:
        result = {'img_name': val_name,
                  'preds': val_pred,
                'labels': val_label}
        pickle.dump(result, f)
            ### --------- ###

    print('test elapsed: {:2.0f}m{:2.0f}s'
          ' test accu: {:.4f}'
          .format(test_elapsed_time // 60, test_elapsed_time % 60, test_accuracy))


print()


def main():
    test_dataset = get_data()

    test_model(test_dataset)


if __name__ == "__main__":
    main()

print('Done')
print()