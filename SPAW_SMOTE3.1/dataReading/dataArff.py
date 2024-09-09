from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt

def loadData(filename):

    data = arff.loadarff(filename)
    df = pd.DataFrame(data[0])
    # 样本
    sample = df.values[:, 0:len(df.values[0]) - 1]
    print(sample)
    # 对标签进行处理
    # [b'1' b'-1' ...]bytes类型
    label = df.values[:, -1]  # 要处理的标签
    cla = []  # 处理后的标签

    for i in label:
        test = int(i)
        cla.append(test)
    X = []
    Y = []
    for i in range(len(sample)):
        X.append(sample[i][0])
        Y.append(sample[i][1])

    return X,Y,cla
    # print (type(sample))
    # print(cla)
    # plt.plot(X, Y, 'go')
    # plt.show()

if __name__ == '__main__':
    filepath = 'D:\\PycharmObject\\dataset\\test\\2d-3c.arff'
    Xdata,Ydata,flag = loadData(filepath)
    print(flag)
    plt.plot(Xdata, Ydata, 'go')
    plt.show()


