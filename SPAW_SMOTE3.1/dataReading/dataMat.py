import scipy.io as sio
import matplotlib.pyplot as plt
def loadData(filename):
    data_train = sio.loadmat(filename)  # 读出的数据是字典

    data_train_label = data_train.get('label')  # 取出字典里的label
    print(data_train_label)
    data_train_data = data_train.get('data')  # 取出字典里的data
    print(data_train_data)
    cla=[]
    for i in data_train_label:
        test = int(i)
        cla.append(test)
    X = []
    Y = []
    for i in range(len(data_train_data)):
        X.append(data_train_data[i][0])
        Y.append(data_train_data[i][1])

    return X, Y, cla

if __name__ == '__main__':
    filepath = "D://PycharmObject//dataset//test//compound.mat"
    Xdata,Ydata,flag = loadData(filepath)
    print(flag)
    plt.plot(Xdata, Ydata, 'go')
    plt.show()

