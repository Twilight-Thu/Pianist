import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
import matplotlib.pyplot as plt

def show_accuracy(predictLabel,Label):
    Label = np.ravel(Label).tolist()
    predictLabel = predictLabel.tolist()
    error = 0
    for i in range(len(Label)):
        error += abs(Label[i] - predictLabel[i])
    return (round(error/len(Label),5))

class node_generator(object):
    def __init__(self, isenhance = False):
        self.Wlist = []
        self.blist = []
        self.function_num = 0
        self.isenhance = isenhance

    def sigmoid(self, x):
        return 1.0/(1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(x, 0)

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def linear(self, x):
        return x

    def orth(self, W):
        for i in range(0, W.shape[1]):
            w = np.mat(W[:,i].copy()).T
            w_sum = 0
            for j in range(i):
                wj = np.mat(W[:,j].copy()).T
                w_sum += (w.T.dot(wj))[0,0]*wj
            w -= w_sum
            w = w/np.sqrt(w.T.dot(w))
            W[:,i] = np.ravel(w)
        return W

    def generator(self, shape, times):
        for i in range(times):
            W = 2*np.random.random(size=shape)-1
            if self.isenhance == True:
                W = self.orth(W)  
            b = 2*np.random.random() -1
            yield (W, b)

    def generator_nodes(self, data, times, batchsize, function_num):
        self.Wlist = [elem[0] for elem in self.generator((data.shape[1], batchsize), times)]
        self.blist = [elem[1] for elem in self.generator((data.shape[1], batchsize), times)]

        self.function_num = {'linear':self.linear,
                        'sigmoid': self.sigmoid,
                        'tanh':self.tanh,
                        'relu':self.relu }[function_num]
        nodes = self.function_num(data.dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            nodes = np.column_stack((nodes, self.function_num(data.dot(self.Wlist[i])+self.blist[i])))
        return nodes	

    def transform(self,testdata):
        testnodes = self.function_num(testdata.dot(self.Wlist[0])+self.blist[0])
        for i in range(1,len(self.Wlist)):
            testnodes = np.column_stack((testnodes, self.function_num(testdata.dot(self.Wlist[i])+self.blist[i])))
        return testnodes

class scaler:
    def __init__(self):
        self._mean = 0
        self._std = 0
    
    def fit_transform(self,traindata):
        self._mean = traindata.mean(axis = 0)
        self._std = traindata.std(axis = 0)
        return (traindata-self._mean)/(self._std+0.001)
    
    def transform(self,testdata):
        return (testdata-self._mean)/(self._std+0.001)

class broadNet(object):
    def __init__(self, map_num=10,enhance_num=10,map_function='linear',enhance_function='linear',batchsize='auto'):
        self.map_num = map_num
        self.enhance_num = enhance_num
        self.batchsize = batchsize
        self.map_function = map_function
        self.enhance_function = enhance_function

        self.W = 0
        self.pseudoinverse = 0
        self.normalscaler = scaler()
        self.onehotencoder = preprocessing.OneHotEncoder(sparse = False)
        self.mapping_generator = node_generator()
        self.enhance_generator = node_generator(isenhance = True)
        

    def fit(self, data, label):
        if self.batchsize == 'auto':
            self.batchsize = data.shape[1]
            print(self.batchsize)

        data = self.normalscaler.fit_transform(data)
        label = self.onehotencoder.fit_transform(np.asarray(label).reshape(-1,1))

        mappingdata = self.mapping_generator.generator_nodes(data, self.map_num, self.batchsize,self.map_function)
        enhancedata = self.enhance_generator.generator_nodes(mappingdata, self.enhance_num, self.batchsize,self.enhance_function)

        print('number of mapping nodes {0}, number of enhence nodes {1}'.format(mappingdata.shape[1],enhancedata.shape[1]))
        print('mapping nodes maxvalue {0} minvalue {1} '.format(round(np.max(mappingdata),5),round(np.min(mappingdata),5)))
        print('enhence nodes maxvalue {0} minvalue {1} '.format(round(np.max(enhancedata),5),round(np.min(enhancedata),5)))

        inputdata = np.column_stack((mappingdata, enhancedata))
        print('input shape ', inputdata.shape)
        pseudoinverse = np.linalg.pinv(inputdata)
        print('pseudoinverse shape:', pseudoinverse.shape)
        self.W = pseudoinverse.dot(label)
        return self.W

    def decode(self,Y_onehot):
        Y = []
        for i in range(Y_onehot.shape[0]):
            lis = np.ravel(Y_onehot[i,:]).tolist()
            Y.append(lis.index(max(lis)))
        return np.array(Y)

    def accuracy(self,predictlabel,label):
        label = np.ravel(label).tolist()
        predictlabel = predictlabel.tolist()
        error = 0
        for i in range(len(label)):
            error += abs(label[i] - predictlabel[i])
        return (round(error/len(label),5))

    def predict(self, testdata):
        testdata = self.normalscaler.transform(testdata)
        test_mappingdata = self.mapping_generator.transform(testdata)
        test_enhancedata = self.enhance_generator.transform(test_mappingdata)

        test_inputdata = np.column_stack((test_mappingdata,test_enhancedata))    
        return self.decode(test_inputdata.dot(self.W))   

if __name__ == '__main__':
    data = pd.read_csv('./results_final.csv')  
  
    le = preprocessing.LabelEncoder()
    for item in data.columns:
        data[item] = le.fit_transform(data[item])

    """
    1、读学长excel，处理出一组测试集xueqing，并验证image_rotates中是否都有（因为这个涉及到对应的问题！！！）
    2、遍历image_rotates，如果该图片在xueqing中，则加入测试集test
    3、遍历image_rotates，不在test中的都加入train
    4、将train、test中的图片分别处理成3072维特征
    5、写入excel的时候加上对应的PM2.5值
    ps：4-5分别对应两个excel
    """
    label = data['PM'].values
    data = data.drop('PM',axis=1)
    data = data.values
    print(data.shape,max(label)+1)

    traindata,testdata,trainlabel,testlabel = train_test_split(data,label,test_size=0.2,random_state = 0)
     #处理训练集，训练集现在不包含文件名，仅包含如此格式的数据：第一列是label，后3072列是feature
    print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape)


    bls = broadNet(map_num = 10, #mapping nodes的数量, 可以改，理论上数量越大精度越好，不能防止过拟合
               enhance_num = 10, #enhance nodes的数量，可以改，理论上数量越大精度越好，不能防止过拟合
               map_function = 'linear',
               enhance_function = 'linear',#激活函数，最好不动，动了不知道会发生啥
               batchsize = 64) #batchsize，有可能调成128效果会更好，代价是训练时间大幅增加

    starttime = datetime.datetime.now()
    Weight = bls.fit(traindata,trainlabel)
    #学习本身，可以理解成黑盒，中间不好动
    endtime = datetime.datetime.now()
    print('the training time of BLS is {0} seconds'.format((endtime - starttime).total_seconds()))

    predictlabel = bls.predict(testdata) 
    #对测试集进行判断，testdata的格式就是48*48*3=3072个feature，对每个样本输出一个浮点数
    print(show_accuracy(predictlabel,testlabel))
