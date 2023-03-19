import torch
from torch import nn

class BPNNModel_DNN(torch.nn.Module):
    def __init__(self):
        # 调用父类的初始化函数，必须要的
        super(BPNNModel_DNN, self).__init__()
        # 创建四个Sequential对象，Sequential是一个时序容器，将里面的小的模型按照序列建立网络
        self.layer1 = nn.Sequential(nn.Linear(6, 2), nn.Sigmoid())
        self.layer2 = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid())
        self.layer3 = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid())
        self.layer4 = nn.Sequential(nn.Linear(2, 1))

    def forward(self, img):
        # 每一个时序容器都是callable的，因此用法也是一样。
        img = img.type(torch.float32)
        img = self.layer1(img)
        img = self.layer2(img)
        img = self.layer3(img)
        img = self.layer4(img)
        return img

class BPNNModel_BNN(torch.nn.Module):
    def __init__(self):
        # 调用父类的初始化函数，必须要的
        super(BPNNModel_BNN, self).__init__()
        # 创建四个Sequential对象，Sequential是一个时序容器，将里面的小的模型按照序列建立网络
        self.layer1 = nn.Sequential(nn.Linear(6, 6), nn.Sigmoid())
        self.layer4 = nn.Sequential(nn.Linear(6, 1))

    def forward(self, img):
        # 每一个时序容器都是callable的，因此用法也是一样。
        img = self.layer1(img)
        img = self.layer2(img)
        img = self.layer3(img)
        img = self.layer4(img)
        return img