# update Roi_cropper
import os
import torch
import random
import numpy as np
import pandas as pd

from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# read the image file
def loadTrainData(image_size:tuple=(28,28), showExample=False):
    data = []
    label = []
    classes = 43
    project_path = os.path.abspath(os.path.join(os.getcwd(), '.')) + "\\GTSRB"

    for i in range(classes):
        path = os.path.join(project_path, "Training\\{:05d}".format(i))
        images = os.listdir(path)
        # print(path)
        for a in images:
            try:
                _image = Image.open(path + "\\" + a)
                _image = _image.resize(image_size)
                _image = np.array(_image)
                data.append(_image)
            except:
                _csv = pd.read_csv(path + "\\" + a, sep=';', encoding="utf-8")
                label = label + _csv['ClassId'].values.tolist()

    print("data={}, label={}, trainingset={}".format(len(data), len(label), len(_csv)))

    if showExample:
        _csv.head()
        plt.imshow(data[0]), plt.title("Example:label[{}]:".format(label[0]))
    
    return data, label

def loadTrainData_Roi(image_size:tuple=(28,28), showExample=False):
    data = []
    label = []
    classes = 43
    project_path = os.path.abspath(os.path.join(os.getcwd(), '.')) + "\\GTSRB"
    _average_size = [0,0]
    
    for i in range(classes):
        path = os.path.join(project_path, "Training\\{:05d}".format(i))
        images = os.listdir(path)
        
        _csv = pd.read_csv(path + "\\" + images[len(images)-1], sep=';', encoding="utf-8")
        label = label + _csv['ClassId'].values.tolist()
        _Roi_X1 = _csv['Roi.X1'].values.tolist()
        _Roi_Y1 = _csv['Roi.Y1'].values.tolist()
        _Roi_X2 = _csv['Roi.X2'].values.tolist()
        _Roi_Y2 = _csv['Roi.Y2'].values.tolist()
        _sum_size = [0,0]

        # print(path)
        for i in range(len(images)):
            try:
                _image = Image.open(path + "\\" + images[i])
                _image = _image.crop((_Roi_X1[i],_Roi_Y1[i],_Roi_X2[i],_Roi_Y2[i]))
                _sum_size[0] = _sum_size[0] + _Roi_X2[i] - _Roi_X1[i]
                _sum_size[1] = _sum_size[1] + _Roi_Y2[i] - _Roi_Y1[i]

                _image = _image.resize(image_size)
                _image = np.array(_image)
                data.append(_image)
            except:
                _average_size[0] = (_average_size[0] + _sum_size[0]/len(_sum_size))/2
                _average_size[1] = (_average_size[1] + _sum_size[1]/len(_sum_size))/2

    print("data={}, label={}, trainingset={}, average_size={}".format(len(data), len(label), len(_csv), _average_size))

    if showExample:
        _csv.head()
        plt.imshow(data[0]), plt.title("Example:label[{}]:".format(label[0]))
    
    return data, label

def seperateDataset(data:list, label:list):
    # shuffle the dataset
    randnum = random.randint(0,100)
    random.seed(randnum)
    random.shuffle(data)
    random.seed(randnum)
    random.shuffle(label)

    # Converting lists into numpy arrays
    data = np.array(data)
    label = np.array(label)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

    # Array -> Tesor
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    # X_train = X_train.unsqueeze(1)
    # X_test = X_test.unsqueeze(1)

    # Tensor转置 -> nn.Conv2d Tensor(batch_size,channels,height,width)
    X_train = X_train.permute(0, 3, 1, 2)
    X_test = X_test.permute(0, 3, 1, 2)

    print("X_train={}, X_test={}\ny_train={}, y_test={}".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

    return X_train, X_test, y_train, y_test

def loadTestData(image_size:tuple=(28,28), showExample=False):
    test = []
    path = os.path.abspath(os.path.join(os.getcwd(), '.')) + "\\GTSRB\\Final_Test\\Images"
    images = os.listdir(path)
    for a in images:
        try:
            _image = Image.open(path + "\\" + a)
            _image = _image.resize(image_size)
            _image = np.array(_image)
            test.append(_image)
        except:
            _csv = pd.read_csv(path + "\\" + a, sep=';', encoding="utf-8")

    print("test images={}".format(len(test)))
    test = torch.FloatTensor(test)
    test = test.permute(0, 3, 1, 2)

    if showExample:
        _csv.head()
        plt.imshow(test[0]), plt.title("Example: test No.1:")
    
    return test

def loadTestData_Roi(image_size:tuple=(28,28), showExample=False):
    test = []
    path = os.path.abspath(os.path.join(os.getcwd(), '.')) + "\\GTSRB\\Final_Test\\Images"
    images = os.listdir(path)
    _average_size = [0,0]

    _csv = pd.read_csv(path + "\\" + images[len(images)-1], sep=';', encoding="utf-8")
    label = label + _csv['ClassId'].values.tolist()
    _Roi_X1 = _csv['Roi.X1'].values.tolist()
    _Roi_Y1 = _csv['Roi.Y1'].values.tolist()
    _Roi_X2 = _csv['Roi.X2'].values.tolist()
    _Roi_Y2 = _csv['Roi.Y2'].values.tolist()
    
    for i in range(len(images)):
        try:
            _image = Image.open(path + "\\" + images[i])
            _image = _image.crop((_Roi_X1[i],_Roi_Y1[i],_Roi_X2[i],_Roi_Y2[i]))
            _average_size[0] = _average_size[0] + _Roi_X2[i] - _Roi_X1[i]
            _average_size[1] = _average_size[1] + _Roi_Y2[i] - _Roi_Y1[i]

            _image = _image.resize(image_size)
            _image = np.array(_image)
            test.append(_image)
        except:
            _average_size[0] = _average_size[0]/len(_average_size)
            _average_size[1] = _average_size[1]/len(_average_size)
    
    print("test images={}, average_size={}".format(len(test), _average_size))
    test = torch.FloatTensor(test)
    test = test.permute(0, 3, 1, 2)

    if showExample:
        _csv.head()
        plt.imshow(test[0]), plt.title("Example: test No.1:")
    
    return test

def predict(model, test, showExample=False):
    # 将预测结果写入txt文件中
    with open("predict_labels_1120222198_张英祺.txt", 'w') as f:
        with torch.no_grad():
            out = model(test)
            _, pred = torch.max(out.data, 1)
        for res in pred.tolist():
            f.write(str(res)+'\n')
    print("Finished Predicting! processed images={}".format(len(pred)))

    if showExample:
        path = os.path.abspath(os.path.join(os.getcwd(), '.')) + "\\GTSRB\\Final_Test\\Images"
        images = os.listdir(path)
        for i in range(10):
            print("{} is labeled as {}".format(images[i], pred[i]))

    return None

def illustrate(loss:list, acc:list, saveFig:str=""):
    _x1 = []
    _x2 = []
    for i in range(1,len(loss)+1):
        _x1.append(i)
        if i % (len(loss)/len(acc)) == 0: _x2.append(i)

    plt.figure(figsize=(20,10))
    ax=plt.axes()
    ax.plot(_x1, loss, marker='x', linestyle = ':', lw=2, label='q=8')
    ax.plot(_x2, acc, marker='o', linestyle = '-', lw=2, label='q=12')
    if len(saveFig)!=0: plt.savefig(saveFig)

    plt.show()
    return None