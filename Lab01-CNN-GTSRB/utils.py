import os
from PIL import Image
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# read the image file
def loadTrainData(showExample=False):
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
                image = Image.open(path + "\\" + a)
                image = image.resize((30,30))
                image = np.array(image)
                data.append(image)
            except:
                csv = pd.read_csv(path + "\\" + a, sep=';', encoding="utf-8")
                label = label + csv['ClassId'].values.tolist()

    print("data={}, label={}, trainingset={}".format(len(data), len(label), len(csv)))

    if showExample:
        csv.head()
        plt.imshow(data[0]), plt.title("Example:label[{}]:".format(label[0]))
    
    return data, label

def seperateDataset(data:list, label:list):
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

def loadTestData(showExample=False):
    test = []
    path = os.path.abspath(os.path.join(os.getcwd(), '.')) + "\\GTSRB\\Final_Test\\Images"
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + "\\" + a)
            image = image.resize((30,30))
            image = np.array(image)
            test.append(image)
        except:
            csv = pd.read_csv(path + "\\" + a, sep=';', encoding="utf-8")

    print("test images={}".format(len(test)))
    test = torch.FloatTensor(test)
    test = test.permute(0, 3, 1, 2)

    if showExample:
        csv.head()
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
    x1 = []
    x2 = []
    for i in range(1,len(loss)+1):
        x1.append(i)
        if i % (len(loss)/len(acc)) == 0: x2.append(i)

    plt.figure(figsize=(20,10))
    ax=plt.axes()
    ax.plot(x1, loss, marker='x', linestyle = ':', lw=2, label='q=8')
    ax.plot(x2, acc, marker='o', linestyle = '-', lw=2, label='q=12')
    if len(saveFig)!=0: plt.savefig(saveFig)

    plt.show()
    return None