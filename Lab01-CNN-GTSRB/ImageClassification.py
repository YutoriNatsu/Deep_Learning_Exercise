import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim


# 数据加载
class GTSRBDataset(Dataset):
    def __init__(self):
        # 添加数据集的初始化内容
        pass

    def __getitem__(self, index):
        # 添加getitem函数的相关内容
        pass

    def __len__(self):
        # 添加len函数的相关内容
        pass


# 构建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义模型的网络结构
        pass

    def forward(self, x):
        # 定义模型前向传播的内容
        pass


# 定义 train 函数
def train():
    # 参数设置
    epoch_num = 10 #训练轮次数
    val_num = 2 #训练几轮验证一次

    for epoch in range(epoch_num): 
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            # Forward

            # Backward

            # Update

            # 可以从训练集中抽取一部分作为验证集，在训练过程中进行验证
            if epoch % val_num == 0:
                validation()

    print('Finished Training!')


# 定义 validation 函数
def validation():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dev_loader:
            images, labels = data
            # 验证部分的内容
            pass

    print("验证集数据总量：", total, "预测正确的数量：", correct)
    print("当前模型在验证集上的准确率为：", correct / total)


# 定义 test 函数
def test():
    # 将预测结果写入txt文件中
    pass


if __name__ == "__main__":
    # 构建数据集
    train_set = GTSRBDataset()
    dev_set = GTSRBDataset()
    test_set = GTSRBDataset()

    # 构建数据加载器
    train_loader = DataLoader(dataset=train_set)
    dev_loader = DataLoader(dataset=dev_set)
    test_loader = DataLoader(dataset=test_set)

    # 初始化模型对象
    net = Net()

    # 定义损失函数
    criterion = None

    # 定义优化器
    optimizer = None

    # 模型训练
    train()

    # 对模型进行测试，并生成预测结果
    test()
