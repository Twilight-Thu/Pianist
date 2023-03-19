# -*- coding: utf-8 -*
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Net import BPNNModel_DNN
from Utils import get_img_info, get_dataAndLabel, ImgDataset

MAX_EPOCH = 10000
LR = 0.0001

train_dir = r"E:\dachuang\Photo_Beijing\daytime\train"
valid_dir = r"E:\dachuang\Photo_Beijing\daytime\valid"

train_data = ImgDataset(data_dir=train_dir)
valid_data = ImgDataset(data_dir=valid_dir)

# construct DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use device :{}".format(device))
# 创建和实例化一个整个模型类的对象
model = BPNNModel_DNN()
model.to(device)
# 打印出整个模型
# print(model)

# Step 3:============================定义损失函数和优化器===================
criterion = nn.MSELoss()
criterion.to(device)
# 我们优先使用随机梯度下降，lr是学习率:
optimizer = torch.optim.SGD(model.parameters(), LR)

# Step 4:============================开始训练网络===================
# 为了实时观测效果，我们每一次迭代完数据后都会，用模型在测试数据上跑一次，看看此时迭代中模型的效果。
# 用数组保存每一轮迭代中，训练的损失值和精确度，也是为了通过画图展示出来。
train_losses = []
train_acces = []
# 用数组保存每一轮迭代中，在测试数据上测试的损失值和精确度，也是为了通过画图展示出来。
eval_losses = []
eval_acces = []

for e in range(MAX_EPOCH):
    # 4.1==========================训练模式==========================
    train_loss = 0
    train_acc = 0
    model.train()   # 将模型改为训练模式
    # 每次迭代都是处理一个小批量的数据，batch_size是64
    for im, label in train_loader:
        #im = torch.tensor(im, dtype=torch.float32)
        #label = torch.tensor(label, dtype=torch.float32)
        im = im.type(torch.float32)
        label = label.type(torch.float32)
        im, label = im.to(device), label.to(device)
        # 计算前向传播，并且得到损失函数的值
        out = model(im)
        loss = criterion(out, label)
        # 反向传播，记得要把上一次的梯度清0，反向传播，并且step更新相应的参数。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        print('epoch: {}, Train Loss Sum: {:.6f}'
              .format(e, train_loss))
    train_losses.append(train_loss / len(train_loader))
    # 4.2==========================每次进行完一个训练迭代，就去测试一把看看此时的效果==========================
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    model.eval()  # 将模型改为预测模式
    # 每次迭代都是处理一个小批量的数据，batch_size是128
    for im, label in valid_loader:
        im, label = im.to(device), label.to(device)
        im = Variable(im)  # torch中训练需要将其封装即Variable，此处封装像素即784
        label = Variable(label)  # 此处为标签
        out = model(im)  # 经网络输出的结果
        # label = label.unsqueeze(1)
        loss = criterion(out, label)  # 得到误差
        # 记录误差
        eval_loss += loss.item()
    eval_losses.append(eval_loss / len(valid_loader))
    # eval_acces.append(eval_acc / len(test_data))
    print('epoch: {}, Train Loss: {:.6f},Eval Loss: {:.6f}'
          .format(e, train_loss / len(train_loader),eval_loss / len(valid_loader)))
plt.title('train loss')
plt.plot(np.arange(len(train_losses)), train_losses)
plt.show()
plt.plot(np.arange(len(train_acces)), train_acces)
plt.title('train acc')
plt.show()
plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.title('test loss')
plt.show()
plt.plot(np.arange(len(eval_acces)), eval_acces)
plt.title('test acc')
plt.show()
# for i in range(10):
#     out = model(x[i, :].to(device))
#     print("predict:","   ",out.detach().numpy())
