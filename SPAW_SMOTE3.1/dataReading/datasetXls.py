#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#读取工作簿和工作簿中的工作表

def loadData(filepath):
    data_frame=pd.read_excel(filepath,sheet_name='Data', header=0, usecols="C:K")
    label_frame=pd.read_excel(filepath,sheet_name='Data', header=0, usecols="B")
    #新建一个工作簿
    data = []
    label = []
    #print(data_frame)
    for indexs in data_frame.index:
        data.append(indexs)
        data[indexs]=[]
        t = len(data_frame.loc[indexs].values[:])
        for i in range(t):
            data[indexs].append(data_frame.loc[indexs].values[i])
    for indexs in label_frame.index:
        label.append(label_frame.loc[indexs].values[0])
        data = np.array(data)

    return data,label

def loadDataMid(filepath):
    data_frame=pd.read_excel(filepath, header=0)
    #新建一个工作簿
    data = []
    label = []
    #print(data_frame)
    for indexs in data_frame.index:
        data.append(indexs)
        data[indexs]=[]
        t = len(data_frame.loc[indexs].values[:])
        for i in range(t):
            data[indexs].append(data_frame.loc[indexs].values[i])
        # data = np.array(data)

    return data

def writeData(list):
    # 二维list
    n = len(list[0])
    s = np.arange(n)
    s.tolist()
    # list转dataframe
    df = pd.DataFrame(list, columns=s)
    # 保存到本地excel
    df.to_excel("D:\\PycharmObject\\annoy\\mid_result\\result.xlsx", index=False)

def writeDataTest(list,num):
    # 二维list
    n = len(list[0])
    s = np.arange(n)
    s.tolist()
    # list转dataframe
    df = pd.DataFrame(list, columns=s)
    # 保存到本地excel
    df.to_excel("D:\\PycharmObject\\annoy\\mid_result\\dataProssesing"+str(num)+".xlsx", index=False)

def writeDataReTest(list, filename):
    # 二维list
    max_length = 0
    for i in range(len(list)):
        if len(list[i]) > max_length:
            max_length = len(list[i])

    s = np.arange(max_length)
    s.tolist()
    # list转dataframe
    df = pd.DataFrame(list, columns=s)
    # 保存到本地excel
    df.to_excel("D:\\PycharmObject\\annoy\\mid_result\\"+str(filename)+".xlsx", index=False)

def write_data_evaluation(list,filename):
    # 二维list
    n = len(list[0])
    s = np.arange(n)
    s.tolist()
    # list转dataframe
    df = pd.DataFrame(list, columns=s)
    # 保存到本地excel
    df.to_excel(filename, index=False)


if __name__ == '__main__':
    filepath = 'D:\\PycharmObject\\dataset\\testhigh\\BreastTissue.xls'
    a, b = loadData(filepath)
    writeData(a)





