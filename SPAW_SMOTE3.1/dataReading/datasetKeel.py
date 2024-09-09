
import numpy as np

def loadData(filepath):
    raw_dataset = np.loadtxt(filepath, delimiter=', ', dtype=str)
    #label = raw_dataset[:,0]
    data_file = raw_dataset[:,0:len(raw_dataset)-1]
    dim = len(data_file[0])
    #print((raw_dataset))
    label = data_file[:, dim-1]
    label.tolist()
    label_num = []
    min_num = 0
    for i in range(len(label)):
        if label[i] == 'negative':
            label_num.append(0)
        else:
            label_num.append(1)
            min_num = min_num + 1
    data = data_file[:, 0:dim-1]
    print("IR is {}".format((len(data)- min_num)/min_num))
    return data.tolist(), label_num


if __name__ == '__main__':
    filepath = 'D:\\PycharmObject\\annoy\\dataset\\KEEL\\glass1.dat'
    data, label= loadData(filepath)
    print(data)
    print(label)
    #data,flag = loadData(filepath)
    #print(flag)
