
import numpy as np

def loadData(filepath):
    raw_dataset = np.loadtxt(filepath, delimiter=',', dtype=str)
    label = raw_dataset[:,0]
    data = raw_dataset[:,1:len(raw_dataset)-1]
    #print((raw_dataset))
    return data.tolist(), label.tolist()


if __name__ == '__main__':
    filepath = 'D:\\PycharmObject\\annoy\\dataset\\UCI\\abalone.data'
    data, label = loadData(filepath)
    print(data)
    print(label)
    #data,flag = loadData(filepath)
    #print(flag)
