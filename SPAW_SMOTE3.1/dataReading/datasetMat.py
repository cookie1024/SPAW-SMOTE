import scipy.io as sio
import matplotlib.pyplot as plt
def loadData(filename):
    data_train = sio.loadmat(filename)  # 读出的数据是字典

    data_train_label = data_train.get('label')  # 取出字典里的label
    #print(data_train_label)
    data_train_data = data_train.get('data')  # 取出字典里的data
    #print(data_train_data)
    cla=[]
    for i in data_train_label:
        test = int(i)
        cla.append(test)

    return data_train_data, cla

if __name__ == '__main__':
    filepath = "D://PycharmObject//dataset//test//compound.mat"
    a,b=loadData(filepath)
    print(b)


