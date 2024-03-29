# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 20:35:24 2018


@author: HAN_RUIZHI yb77447@umac.mo OR  501248792@qq.com

This code is the first version of BLS Python.
If you have any questions about the code or find any bugs
   or errors during use, please feel free to contact me.
If you have any questions about the original paper,
   please contact the authors of related paper.
"""

'''

Installation has been tested with Python 3.5.
Since the package is written in python 3.5, 
python 3.5 with the pip tool must be installed first. 
It uses the following dependencies: numpy(1.16.3), scipy(1.2.1), keras(2.2.0), sklearn(0.20.3)  
You can install these packages first, by the following commands:

pip install numpy
pip install scipy
pip install keras (if use keras data_load())
pip install scikit-learn
'''
import torch
import csv
from snntorch import spikegen
from sklearn import preprocessing
import numpy as np
import scipy.io as scio
from BroadLearningSystem import BLS, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

''' For Keras dataset_load()'''


labeldataFile = 'F:\\EAAIBCI\\Trainingset\\Data_Sample01.mat'
BCIdataFile = 'F:\\EAAIBCI\\Features\\Normalizedfeatures1.csv'
labeldata = scio.loadmat(labeldataFile)
traindata = labeldata['epo_train']
label = np.double(traindata[0][0][5])
label = np.transpose(label)

BCIdata = []
with open(BCIdataFile, "r") as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        BCIdata.append(row)

# 关闭文档
csvfile.close()
BCIdata = np.array(BCIdata)
BCIdata = BCIdata.astype(float)
print(BCIdata.shape)

labels_indices = np.argmax(label, axis=1)
# 使用相同的索引对标签和特征数组进行排序
sorted_indices = np.argsort(labels_indices)
sorted_labels = label[sorted_indices]
sorted_features = BCIdata[sorted_indices]
# 先划分训练集和测试集
traindata, testdata, trainlabel, testlabel = train_test_split(sorted_features, sorted_labels, test_size=0.1,)

# 再由训练集划分训练和验证集

train_X1, train_X2, train_y1, train_y2 = train_test_split(traindata, trainlabel, test_size=0.33, random_state=1)
train_X4, train_X3, train_y4, train_y3 = train_test_split(train_X1, train_y1, test_size=0.5, random_state=1)

N1 =10 #  # of nodes belong to each window
N2 =10 #  # of windows -------Feature mapping layer
N3 =100  # of enhancement nodes -----Enhance layer
L = 5    #  # of incremental steps
M1 = 30  #  # of adding enhance nodes
s = 0.8  #  shrink coefficient
C = 2**-30 # Regularization coefficient

print('-------------------BLS_BASE---------------------------')
B1,B1test,B1test1,test_acc1=BLS(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3)
# B1,B1test,B1test1=BLS(train_X2, train_y2, testdata, testlabel, s, C, N1, N2, N3)
# B2,B2test,B2test1=BLS(train_X3,train_y3, testdata, testlabel, s, C, N1, N2, N3)
# B3,B3test,B3test1=BLS(train_X4,train_y4, testdata, testlabel, s, C, N1, N2, N3)
# B4,B4test,B4test1=BLS(train_X2,train_y2, testdata, testlabel, s, C, N1, N2, N3)
# B5,B5test,B5test1=BLS(train_X2,train_y2, testdata, testlabel, s, C, N1, N2, N3)
# # B6,B6test,B6test1=BLS(train_X2,train_y2, testdata, testlabel, s, C, N1, N2, N3)
# # print(type(B1))
# outputweight1=np.double(np.vstack((B1,B2,B3,B4,B5)))
# InputOfOutputLayerTest1=np.double(np.hstack((B1test,B2test,B3test,B4test,B5test)))
# # print(outputweight1.shape)
# # print(InputOfOutputLayerTest1.shape)
# OutputOfTest = np.dot(InputOfOutputLayerTest1,outputweight1)
# def show_accuracy(predictLabel, Label):
#     count = 0
#     label_1 = np.zeros(Label.shape[0])
#
#     label_1 = Label.argmax(axis=1)
#     predlabel = []
#     predlabel = predictLabel.argmax(axis=1)
# #    predlabel = torch.topk(predlabel, 1)[1].squeeze(1)
#
#     for j in list(range(Label.shape[0])):
#         if label_1[j] == predlabel[j]:
#             count += 1
#
#     return (round(count/len(Label),5))
# testAcc = show_accuracy(OutputOfTest,testlabel)
# print(testAcc)
#
#
# B1test11=np.zeros(shape=(len(B1test1),B1test1.shape[1]))
# B1test11=np.mat(B1test11)
#
# for i in range (len(B1test1)):
#     for j in range (len(B1test1[0])):
#         B1test11[i][j]=(B1test1[i][j]+B2test1[i][j]+B3test1[i][j])/3
# testAcc1 = show_accuracy(B1test11,testlabel)
# print(testAcc1)









