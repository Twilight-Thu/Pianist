import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import Utils
import math
import numpy as np
import matplotlib.pyplot as plt

from Net import BPNNModel_DNN, BPNNModel_BNN


def get_dataAndLabel(FILE_PATH):
    # 最大最小PM值
    maxv = 262
    minv = 1
    # 存储网络输入信息
    dic = {}
    x = []
    t = []
    # 存储对应信息的label
    y = []
    f = open(FILE_PATH, encoding='gbk')
    # 用来标记应该跳过几行信息，查询的PM是nan的话则标记为2表示后两行信息跳过存储
    flag = 0
    for line in f:
        # 如果不是换行则查询信息
        if (line != '\n'):
            # 当前行是path
            if (line[0] == '/'):
                lineSplit = line.split('/')
                imgName = lineSplit[-1].strip()
                PM = 1
                # 查询出的PM不是nan
                if (math.isnan(PM) == False):
                    # 归一化
                    PM = (PM - minv) / (maxv - minv)
                    y.append([float(PM)])
                # PM查询为nan
                else:
                    flag = 2
                continue
            # 获取暗通道和饱和度信息熵
            if (flag == 0):
                lineSplit = line.split(':')
                t.append(float(lineSplit[-1].strip()))
            else:
                flag = flag - 1
        # 是换行符，则存储，清空临时列表
        else:
            if (len(t)):
                x.append(t)
                dic[imgName] = t;
                t = []
                imgName = ""
    x = np.array(x)
    y = np.array(y)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    return dic


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weight0 = 0.5
    weight1 = 0.5
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(norm_mean, norm_std),
    ])

    # load image
    data_dir = r"/home/D-19-zjk/daytime/valid"
    train_data = Utils.ImgDataset(data_dir=data_dir)
    test_loader = DataLoader(dataset=train_data, batch_size=32)
    data_info = list()
    t = 0
    res_list = []
    for im, label in test_loader:
        im = im.type(torch.float32)
        label = label.type(torch.float32)
        im, label = im.to(device), label.to(device)
        model0 = BPNNModel_DNN()
        model1 = BPNNModel_BNN()
        dnn_weights_path = "/home/D-19-zjk/environment-master/environment_D5.pth"
        bnn_weights_path = "/home/D-19-zjk/environment-master/environment_B5.pth"
        model0.load_state_dict(torch.load(dnn_weights_path, map_location=device))
        model1.load_state_dict(torch.load(bnn_weights_path, map_location=device))
        model0.eval()
        model1.eval()
        with torch.no_grad():
            out0 = model0(im)
            out1 = model1(im)
            out = weight0 * out0 + weight1 * out1
            res = out.item() * 261 + 1
            if (res < 1):
                res = 1
            if (res > 262):
                res = 262

            print("res:'{}',PM:'{}'".format(res, label * 261 + 1))
            res_list.append(res)
            t = t + (res - (label * 261 + 1)) ** 2
            print(t)

    print(t)
    print(len(test_loader))
    t = math.sqrt(t / len(test_loader))
    print(t)

    real = []
    for im, label in test_loader:
        real.append(label)

    plt.scatter(res_list, real, marker='.')
    plt.savefig('./double5_scatter.png')
    plt.show()

    plt.figure()
    x = np.arange(len(data_info))
    plt.plot(x, res_list, color='#FF0000', label='predict', linewidth=3.0)
    plt.plot(x, real, color='#00FF00', label='real', linewidth=3.0)
    plt.savefig('./double5_line.png')
    plt.show()


if __name__ == '__main__':
    main()
