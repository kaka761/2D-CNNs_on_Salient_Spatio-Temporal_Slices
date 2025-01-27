import os


path = './Datasets/AR/testTrainsplits/'
save_path = './Datasets/AR/'
files = os.listdir(path)
total = []
for i in range(len(files)-1):
    file = path+files[i]
    if file.split('_')[-1] == 'split2.txt':
        print(file)
        with open(file, 'r') as f:
            data = f.readlines()
            for j in range(len(data)-1):
                if data[j].split(' ')[1] == '1': # train: 1; test: 2; val: 0
                   total.append([data[j].split(' ')[0]+'\n'])
total = sum(total,[])#
fi = open(save_path+'train2.txt', 'w')
for t in total:
    fi.write(t)
fi.close()