import math
import os
import PIL
import torch
import numpy as np
import pandas as pd
from PIL import Image
from cv2 import cv2
from torch.utils.data import Dataset

#获取图片名及其对应的PM2.5信息
def get_img_info(CSV_PATH):
    DF_DATA = pd.read_csv(CSV_PATH)
    DATA_XYs = DF_DATA[['file_Id', 'PM2.5']].values
    return DATA_XYs

#获取data和label
def get_dataAndLabel(FILE_PATH,dataInfo):
    # 最大最小PM值
    maxv = 262
    minv = 1
    # 存储网络输入信息
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
                PM = dataInfo[imgName]
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
            if (len(t)): x.append(t)
            t = []
    x = np.array(x)
    y = np.array(y)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    return x, y

def DarkChannel(im, win=15):
    """ 求暗通道图
    :param im: 输入 3 通道图
    :param win: 图像腐蚀窗口, [win x win]
    :return: 暗通道图
    """
    if isinstance(im, PIL.Image.Image):
        im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))  # 结构元素
    dark = cv2.erode(dc, kernel)                                    # 腐蚀操作
    return Image.fromarray(dark, mode='L')

def SaturationMap(im):
    im = Image.fromarray(np.uint8(im))
    img_hsv = im.convert("HSV")
    h, s, v = img_hsv.split()  # 分离三通道
    return s

class ImgDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # self.label_name = {"0": 0, "1": 1, "2": 2}
        self.data_info = self.getInfo(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        img = list()
        t = list()
        w = [1, 3, 5]
        path_img, label = self.data_info[index]
        img_ = cv2.imread(path_img)
        dc = DarkChannel(img_)
        saturation = SaturationMap(img_)
        img_ = np.stack([dc, saturation])
        #获取信息熵
        for i in w :
            H1 = self.calc_array(self.process_img(img_[0], i))[0]
            H2 = self.calc_array(self.process_img(img_[1], i))[0]
            img.append(H1)
            img.append(H2)
        t.append(float(label))
        t = torch.from_numpy(np.array(t))
        img = torch.from_numpy(np.array(img))
        return img, t

    def __len__(self):
        return len(self.data_info)

    def calc_array(self,img):
        entropy = []

        hist = cv2.calcHist([img], [0], None, [256], [0, 255])
        total_pixel = img.shape[0] * img.shape[1]

        for item in hist:
            probability = item / total_pixel
            if probability == 0:
                en = 0
            else:
                en = -1 * probability * (np.log(probability) / np.log(2))
            entropy.append(en)

        sum_en = np.sum(entropy)
        return sum_en

    #处理图像，详见论文公式4
    def process_img(self, img_, w):
        img = img_.copy().astype(np.int16)

        c = np.ones_like(img)
        c = c * 255
        t = np.add(w * (np.subtract(img, c)), c)
        img = np.minimum(c, t)

        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    #获取图像数据和标签
    def getInfo(self, data_dir):
        data_info = list()
        img_names = os.listdir(data_dir)
        img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
        DF_DATA = r'E:\dachuang\Photo_Beijing.csv'
        DF_DATA = pd.read_csv(DF_DATA)
        DATA_XYs = DF_DATA[['file_Id', 'PM2.5']].values
        for i in range(len(img_names)):
            img_name = img_names[i]
            path_img = os.path.join(data_dir, img_name)
            label = DATA_XYs[np.where(DATA_XYs[:, 0] == img_name)][-1][-1]
            if (math.isnan(label) == False):
                label = (label - 1) / (262 - 1)  # 归一化
                data_info.append((path_img, label))

        if len(data_info) == 0:
            raise Exception("\n data_dir:{} is a empty dir! Please checkout your path to images!".format(data_dir))
        return data_info