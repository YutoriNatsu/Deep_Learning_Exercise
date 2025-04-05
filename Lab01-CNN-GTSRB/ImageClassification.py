# comlpete
import torch
import torch.nn as nn

# construct CNN model
class Net(nn.Module):
    def __init__(self, image_size:tuple=(28,28), img_channels=3, classes=43):
        super().__init__()

        self.layer1 = nn.Sequential()
        self.layer1.add_module("conv", nn.Conv2d(in_channels = img_channels, 
                                                 out_channels = 32, 
                                                 kernel_size = 5, 
                                                 stride = 1, 
                                                 padding = 2))
        self.layer1.add_module('relu', nn.ReLU())
        self.layer1.add_module('pool', nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.layer2 = nn.Sequential()
        self.layer2.add_module("conv", nn.Conv2d(in_channels = 32, 
                                                 out_channels = 64, 
                                                 kernel_size = 5, 
                                                 stride = 1, 
                                                 padding = 2))
        self.layer2.add_module('relu', nn.ReLU())
        self.layer2.add_module('pool', nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.dropout = nn.Dropout2d(p = 0.5)
        self.flatten = nn.Flatten(start_dim = 1, end_dim = 3)

        self.fc1 = nn.Linear(in_features = 4*image_size[0]*image_size[1], out_features = 1000)

        self.fc2 = nn.Linear(in_features = 1000, out_features = classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # print("conv:{}".format(x.shape))

        x = self.flatten(x)
        x = self.dropout(x)
        # print("flatten:{}".format(x.shape))

        x = self.fc1(x)
        x = self.fc2(x)
        # print("output:{}".format(x.shape))

        return x

def adjust_learning_rate(learning_rate, learning_rate_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR multiplied by learning_rate_decay(set 0.98, usually) every epoch"""
    learning_rate = learning_rate * (learning_rate_decay ** epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    return learning_rate

# define training function
import time
from random import shuffle
def train(x_train, y_train, x_test, y_test, model,
          loss_function, optimizer,
          BATCH_SIZE:int = 64, EPOCH_NUM:int = 40, VAL_NUM:int = 2,
          output_log = False):
    
    train_N = x_train.shape[0]
    loss_rate = []
    acc_rate = []

    # torch.autograd.set_detect_anomaly(True)
    _begin = time.time()
    for epoch in range(1,EPOCH_NUM+1):
        adjust_learning_rate(learning_rate=0.0001,
                             learning_rate_decay=0.98,
                             optimizer=optimizer, epoch=epoch)
        _batchindex = list(range(int(train_N / BATCH_SIZE)))
        shuffle(_batchindex)

        for i in _batchindex:
            batch_x = x_train[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
            batch_y = y_train[i*BATCH_SIZE: (i+1)*BATCH_SIZE]

            y_hat = model(batch_x)
            loss = loss_function(y_hat, batch_y)
            optimizer.zero_grad()
            
            # with torch.autograd.detect_anomaly(): loss.backward()
            loss.backward()
            optimizer.step()
        
        loss_rate.append(loss.item())
        
        # test
        if epoch % VAL_NUM == 0:
            _end = time.time()
            y_hat = model(x_test)
            y_hat = torch.max(y_hat, 1)[1].data.squeeze()
            acc = torch.sum(y_hat == y_test).float() / y_test.shape[0]
            acc_rate.append(acc.item())

            if output_log:
                """
                from logging import basicConfig, DEBUG
                basicConfig(level = DEBUG, filename = 'train.log', filemode='w',
                    format=f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] epoch {epoch} | loss:{loss:.4f} | acc:{acc:.4f} | time:{(_end-_begin):.2f}s"
                    )
                """
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] epoch {epoch} | loss:{loss:.4f} | acc:{acc:.4f} | time:{(_end-_begin):.2f}s")
            
            _begin = time.time()
    
    print("Finished Training! BATCH_SIZE={}, EPOCH={}, VAL_NUM={}".format(BATCH_SIZE, EPOCH_NUM, VAL_NUM))
    return loss_rate, acc_rate

def cnn(x_train, y_train, x_test, y_test):
    BATCH_SIZE = 100
    model = Net()
    loss_function = nn.CrossEntropyLoss() # 多分类任务
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(x_train, y_train, x_test, y_test,
          BATCH_SIZE, model, loss_function, optimizer)