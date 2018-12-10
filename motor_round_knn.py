import numpy as np
from collections import Counter
import pandas as pd
import xlwt
import csv

def creatData_train(n,m,filename): #创建训练数据集
    #dataset = xlrd.open_workbook("C:\Users\hasee\Desktop\InputData.xlsx")
    dataset = pd.read_csv(filename)
    dataset =np.mat(dataset)
    #num=index(max(value.(index(0-10)))
    ExtDataset=np.mat([None]*m)
    dataset1 = dataset[:,1:n]
    maxIndex = np.argmax(dataset1,axis = 1)
    for i in range(dataset.shape[0]):
        subIndex = maxIndex[i,0]                       #求前n个值中最大值的索引,往后延续m列
        subDataset = dataset[i,subIndex:subIndex+m]
        ExtDataset =np.vstack((ExtDataset,subDataset))
    ExtDataset =ExtDataset[1:,:]
    dataset0 = ExtDataset[:,:]
    train_labels =dataset[:,-1]
    return dataset0,train_labels


def classify(inX,dataset_train,labels_train,k):
    labels_train = labels_train
    datasetSize = dataset_train.shape[0]
    diffMat = np.tile(inX,(datasetSize,1))-dataset_train
    sqDiffMat = np.multiply(diffMat,diffMat)  #求每个元素的平方
    sqDistance = sqDiffMat.sum(axis=1)         #横向求和
    sortedDistIndicies = np.argsort(sqDistance,axis=0)  #通过列的大小对索引排序
    sortedDistIndicies = np.mat(sortedDistIndicies)
    #   print(sortedDistIndicies)
    for i in range(k):
        countlabels = []
        num = sortedDistIndicies[i,0]
        countlabels.append(labels_train[num,0])
    subCountlabels =set(countlabels)
    for each_subCountlabels in subCountlabels:
        count=0
        for each_countlabels in countlabels:
            if each_subCountlabels == each_countlabels:
                count+=1
                if count>1:
                    result = Counter(countlabels).most_common(1)[0][0]  #获取前k个数重复次数最多的索引
                else:
                    result =countlabels[0]
    return result


def set_style(name,height,bold):         # 设置单元格格式,字体，单元格高度，是否加粗
    style=xlwt.XFStyle()             # 初始化单元格
    font = xlwt.Font()         # 为样式创建字体
    font.name = name           # 是什么样式的字体
    font.bold = False         # 是否加黑
    font.color_index = 4
    font.height = height     # 高度
    style.font = font
    return style


train_dataset,train_labels = creatData_train(3000,1000,'E:\dataset\Train.csv')
test_dataset,A = creatData_train(3000,1000,'E:\dataset\Test_data.csv')
size = test_dataset.shape[0]

id = []
label = []
for i in range(size):
    inX = test_dataset[i,:]
    k=4
    result = (classify(inX, train_dataset, train_labels, k))
    result =int(result)
    t = i+1
    id.append(t)
    label.append(result)
dataframe = pd.DataFrame({'id': id,'label':label})  #创建字典，‘id’和‘label’为第一行的名称，传入的为字典
dataframe.to_csv("test.csv", index=False, sep=',')

