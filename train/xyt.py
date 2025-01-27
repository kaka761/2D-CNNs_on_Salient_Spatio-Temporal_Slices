import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
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
parser.add_argument('-s', '--seq', default=26, type=int, help='sequence length, default 4')
parser.add_argument('-t', '--train', default=256, type=int, help='train batch size, default 100')
parser.add_argument('-v', '--val', default=256, type=int, help='valid batch size, default 8')
parser.add_argument('-o', '--opt', default=1, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=30, type=int, help='epochs to train and val, default 25')
parser.add_argument('-w', '--work', default=4, type=int, help='num of workers to use, default 2')
parser.add_argument('-f', '--flip', default=0, type=int, help='0 for not flip, 1 for flip, default 0')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
parser.add_argument('-l', '--lr', default=1e-4, type=float, help='learning rate for optimizer, default 1e-3')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
parser.add_argument('--weightdecay', default=0, type=float, help='weight decay for sgd, default 0')
parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd, default 0')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum, default False')
parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
parser.add_argument('--sgdstep', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
parser.add_argument('--sgdgamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')
parser.add_argument('--srate', default=1, type=int, help='sample rate')

parser.add_argument('--save_path', default='./Projects/slice_ar/results/xyt/', type= str, help='results save path')
parser.add_argument('--anno_t', default='./Projects/slice_ar/csv/cam/xyt_train.csv', type=str, help='train annotation path')
parser.add_argument('--anno_v', default='./Projects/slice_ar/csv/cam/xyt_val.csv', type=str, help='val annotation path')

args = parser.parse_args()

gpu_usg = ",".join(list(map(str, args.gpu)))
sequence_length = args.seq
train_batch_size = args.train
val_batch_size = args.val
optimizer_choice = args.opt
multi_optim = args.multi
epochs = args.epo
workers = args.work
use_flip = args.flip
crop_type = args.crop
learning_rate = args.lr
momentum = args.momentum
weight_decay = args.weightdecay
dampening = args.dampening
use_nesterov = args.nesterov

sgd_adjust_lr = args.sgdadjust
sgd_step = args.sgdstep
sgd_gamma = args.sgdgamma
srate = args.srate

save_path = args.save_path
anno_t = args.anno_t
anno_v = args.anno_v


os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()

print('number of gpu   : {:6d}'.format(num_gpu))
print('train batch size: {:6d}'.format(train_batch_size))
print('valid batch size: {:6d}'.format(val_batch_size))
print('optimizer choice: {:6d}'.format(optimizer_choice))
print('multiple optim  : {:6d}'.format(multi_optim))
print('num of epochs   : {:6d}'.format(epochs))
print('num of workers  : {:6d}'.format(workers))
print('test crop type  : {:6d}'.format(crop_type))
print('whether to flip : {:6d}'.format(use_flip))
print('learning rate   : {:.4f}'.format(learning_rate))
print('momentum for sgd: {:.4f}'.format(momentum))
print('weight decay    : {:.4f}'.format(weight_decay))
print('dampening       : {:.4f}'.format(dampening))
print('use nesterov    : {:6d}'.format(use_nesterov))
print('method for sgd  : {:6d}'.format(sgd_adjust_lr))
print('step for sgd    : {:6d}'.format(sgd_step))
print('gamma for sgd   : {:.4f}'.format(sgd_gamma))


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
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
        self.count += 1
        return img.crop((x1, y1, x1 + tw, y1 + th))



class Dataset(Dataset):
    def __init__(self, file_paths, file_paths2, file_paths3, file_labels,
                 transform=None, loader=pil_loader):
        self.file_paths = file_paths
        self.file_paths2 = file_paths2
        self.file_paths3 = file_paths3
        self.file_labels_phase = file_labels
        self.transform = transform
        self.loader = loader
        self.frames = []

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        img_names2 = self.file_paths2[index]
        img_names3 = self.file_paths3[index]
        labels_phase = self.file_labels_phase[index]
        imgs = self.loader(img_names)
        imgs2 = self.loader(img_names2)
        imgs3 = self.loader(img_names3)
        if self.transform is not None:
            imgs = self.transform(imgs)
            imgs2 = self.transform(imgs2)
            imgs3 = self.transform(imgs3)
        return imgs, imgs2, imgs3, labels_phase, img_names.split('/')[-1]
    def __len__(self):
        return len(self.file_paths)


class resnet_lstm(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm, self).__init__() 
        resnet = models.resnet18(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.fc = nn.Linear(3*512, 10)

        #init.xavier_normal_(self.lstm.all_weights[0][0]) 
        #init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x, x1, x2): 
        x = self.share.forward(x)
        x = x.view(-1, 512)
        x1 = self.share.forward(x1)
        x1 = x1.view(-1, 512)
        x2 = self.share.forward(x2)
        x2 = x2.view(-1, 512)
        y = torch.cat([x, x1, x2], dim = 1)
        y = self.fc(y) 
        return y

def get_data():
    df_train = pd.read_csv(anno_t, header = None)
    dft = df_train.values
    train_paths = dft[:, 0]
    train_labels = dft[:, 1]
    train_paths2 = dft[:, 2]
    train_paths3 = dft[:, 3]
    df_val = pd.read_csv(anno_v, header = None)
    dfv = df_val.values
    val_paths = dfv[:, 0]
    val_labels = dfv[:, 1]
    val_paths2 = dfv[:, 2]
    val_paths3 = dfv[:, 3]
    print('train_paths  : {:6d}'.format(len(train_paths)))
    print('train_labels : {:6d}'.format(len(train_labels)))
    print('valid_paths  : {:6d}'.format(len(val_paths)))
    print('valid_labels : {:6d}'.format(len(val_labels)))
    
    '''
    ### for Ver ###
    label_dict = {'D0X':0, 'B0A':1, 'B0B':2, 'G01':3, 'G02':4, 'G03':5, 'G04':6, 'G05':7, 'G06':8, 
                  'G07':9, 'G08':10, 'G09':11, 'G10':12, 'G11':13}
    
    label_dict = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8}
    '''
    label_dict = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9}
    train_labels = [label_dict[l] for l in train_labels]
    val_labels = [label_dict[l] for l in val_labels]
       
    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    ### ------- ### 

    train_transforms = transforms.Compose([
        transforms.Resize([128, 128]),
        RandomCrop(112),
        #transforms.Resize([256, 256]),
        #RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])
    test_transforms = transforms.Compose([
        transforms.Resize([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
    ])

    train_dataset = Dataset(train_paths, train_paths2, train_paths3, train_labels, train_transforms)
    val_dataset = Dataset(val_paths, val_paths2, val_paths3, val_labels, test_transforms)

    return train_dataset, val_dataset



class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


def train_model(train_dataset, val_dataset):
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=workers,
        pin_memory=False
        )
    model = resnet_lstm()
    if use_gpu:
        model = model.cuda()

    model = DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy = 0.0
    correspond_train_acc = 0.0

    t_loss = []
    v_loss = []
    t_acc = []
    v_acc = []

    record_np = np.zeros([epochs, 4])
    for epoch in range(epochs):
        L = []
        P = []
        val_name = []
        val_pred = []
        val_label = []
        np.random.seed(epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle = True,
            num_workers=workers,
            pin_memory=False
            )

        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_start_time = time.time()
        num = 0
        train_num = 0

        for data in train_loader:
            num = num + 1
            inputs, inputs1, inputs2, labels_phase, img_names  = data
            if use_gpu:
                inputs = Variable(inputs.cuda())
                inputs1 = Variable(inputs1.cuda())
                inputs2 = Variable(inputs2.cuda())
                labels = Variable(labels_phase.cuda())
                img_names = img_names
            else:
                inputs = Variable(inputs)
                inputs1 = Variable(inputs1.cuda())
                inputs2 = Variable(inputs2.cuda())
                labels = Variable(labels_phase)
            optimizer.zero_grad()
            outputs = model.forward(inputs, inputs1, inputs2)
            outputs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs.data, 1)
            print(num)
            print(preds)
            print(labels)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.data
            train_corrects += torch.sum(preds == labels.data)
            train_num += labels.shape[0]
            print(train_corrects.cpu().numpy() / train_num)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if train_corrects.cpu().numpy() / train_num > 0.75:
                torch.save(copy.deepcopy(model.state_dict()), save_path +'test.pth')

        train_elapsed_time = time.time() - train_start_time
        train_accuracy = train_corrects.cpu().numpy() / train_num
        train_average_loss = train_loss / train_num
        t_acc.append(train_accuracy)
        t_loss.append(train_average_loss.cpu().numpy())


        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        val_start_time = time.time()
        for data in val_loader:
            inputs, inputs1, inputs2, labels_phase, img_names = data
            if use_gpu:
                inputs = Variable(inputs.cuda())
                inputs1 = Variable(inputs1.cuda())
                inputs2 = Variable(inputs2.cuda())
                labels = Variable(labels_phase.cuda())
                img_names = img_names 
            else:
                inputs = Variable(inputs)
                inputs1 = Variable(inputs1.cuda())
                inputs2 = Variable(inputs2.cuda())
                labels = Variable(labels_phase)

            if crop_type == 0 or crop_type == 1:
                outputs = model.forward(inputs, inputs1, inputs2)
            elif crop_type == 5:
                inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
                inputs = inputs.view(-1, 3, 224, 224)
                outputs = model.forward(inputs)
                outputs = outputs.view(5, -1, 3)
                outputs = torch.mean(outputs, 0)
            elif crop_type == 10:
                inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
                inputs = inputs.view(-1, 3, 224, 224)
                outputs = model.forward(inputs, inputs1, inputs2)
                outputs = outputs.view(10, -1, 3)
                outputs = torch.mean(outputs, 0)

            _, preds = torch.max(outputs.data, 1)
            print(num)
            print(preds)
            print(labels)
            ###
            val_name.append(img_names)
            val_pred.append(preds.cpu().numpy())
            val_label.append(labels.cpu().numpy())
            ###
            L += labels.cpu().data.numpy().tolist()
            P += preds.cpu().data.numpy().tolist()
            loss = criterion(outputs, labels)
            val_loss += loss.data
            val_corrects += torch.sum(preds == labels.data)
            val_num += labels.shape[0]
        val_elapsed_time = time.time() - val_start_time
        val_accuracy = val_corrects.cpu().numpy() / val_num
        val_average_loss = val_loss / val_num
        v_acc.append(val_accuracy)
        v_loss.append(val_average_loss.cpu().numpy())
        print('epoch: {:4d}'
                  ' train in: {:2.0f}m{:2.0f}s'
                  ' train loss: {:4.4f}'
                  ' train accu: {:.4f}'
                  ' valid in: {:2.0f}m{:2.0f}s'
                  ' valid loss: {:4.4f}'
                  ' valid accu: {:.4f}'
                  .format(epoch,
                      train_elapsed_time // 60,
                      train_elapsed_time % 60,
                      train_average_loss,
                      train_accuracy,#TA, #
                      val_elapsed_time // 60,
                      val_elapsed_time % 60,
                      val_average_loss,
                      val_accuracy))#VA)) #
        if optimizer_choice == 0:
            if sgd_adjust_lr == 0:
                exp_lr_scheduler.step()
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler.step(val_average_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            correspond_train_acc = train_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            best_L = L
            best_P = P
            ### save .pkl ###
            file_name = save_path + str(epoch) +'.pkl'
            with open(file_name,'wb') as f:
                result = {'img_name': val_name,
                    'preds': val_pred,
                    'labels': val_label}
                pickle.dump(result, f)
            ### --------- ###
        if val_accuracy == best_val_accuracy:
            if train_accuracy > correspond_train_acc:
                correspond_train_acc = train_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())
        record_np[epoch, 0] = train_accuracy
        record_np[epoch, 1] = train_average_loss
        record_np[epoch, 2] = val_accuracy
        record_np[epoch, 3] = val_average_loss
        np.save(save_path + str(epoch) + '.npy', record_np)

    print('best accuracy: {:.4f} cor train accu: {:.4f}'.format(best_val_accuracy, correspond_train_acc))
    #best_acc += best_val_accuracy
    #cor_acc += correspond_train_acc

    save_val = int("{:4.0f}".format(best_val_accuracy * 10000))
    save_train = int("{:4.0f}".format(correspond_train_acc * 10000))
    model_name = save_path + "best.pth"

    torch.save(best_model_wts, model_name)

    record_name = save_path + "best.npy"
    np.save(record_name, record_np)

    plt.plot(t_loss, label='Training loss')
    plt.plot(v_loss, label='Validation loss')
    plt.title('Loss Metrics')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(save_path +'loss_flod'  + '.png')
    plt.close()

    plt.plot(t_acc, label='Training accuracy')
    plt.plot(v_acc, label='Validation accuracy')
    plt.title('Accuracy Metrics')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(save_path +'acc_flod' + '.png')
    plt.close()

    L = np.asarray(best_L)
    P = np.asarray(best_P)

    arr = confusion_matrix(L, P)
    ConfusionMatrixDisplay(arr).plot()
    plt.savefig(save_path +'confusion_matrix_flod' + '.png')
    plt.close()

    print( f"_Clasification Report\n\n{classification_report(L, P)}")
    
    return best_val_accuracy, correspond_train_acc

def main():
    train_dataset, val_dataset = get_data()
    best_val_accuracy, correspond_train_acc = train_model(train_dataset, val_dataset)
    print('best accuracy: {:.4f} cor train accu: {:.4f}'.format(best_val_accuracy, correspond_train_acc))


if __name__ == "__main__":
    main()

print('Done')
print()