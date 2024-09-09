import numpy as np
from dataset.dataProcessing import data_read_keel
import logging
import itertools
import time
from sklearn.neighbors import NearestNeighbors
from dataReading.datasetArff import loadData

def select_two_label_statistics(y):
    """
    determines class sizes and minority and majority labels
    Args:
    X (np.array): features
    y (np.array): target labels
    """
    unique, counts = np.unique(y, return_counts=True)

    # num = len(y)
    # dataclass = len(unique)
    # key = int(num/dataclass)
    # print(unique)
    # print(counts)
    # print(key)
    if counts[0] > counts[1]:
        min_label = unique[1]
        maj_label = unique[0]
    else:
        min_label = unique[0]
        maj_label = unique[1]

    return min_label, maj_label


def select_label_statistics(y):
    """
    determines class sizes and minority and majority labels
    Args:
    X (np.array): features
    y (np.array): target labels
    """
    unique, counts = np.unique(y, return_counts=True)

    num = len(y)
    dataclass = len(unique)
    key = int(num/dataclass)
    # print(unique)
    # print(counts)
    # print(key)
    min_label = []
    maj_label = []
    for i in range(dataclass):
        if counts[i] < key:
            min_label.append(unique[i])
        else:
            maj_label.append((unique[i]))
    return min_label, maj_label

def check_enough_min_samples_for_sampling(min_label, threshold=2):
    if min_label < threshold:
        print("The number of minority samples (%d) is not enough "
                "for sampling".format(min_label))
        return False
    return True


if __name__ == '__main__':
        filepath = 'D:\\PycharmObject\\ZQannoy3.1\\dataset\\KEEL\\iris0.dat'
        data_ori, tag_ori = data_read_keel(filepath)
        print(tag_ori)
        min_label, maj_label = select_label_statistics(tag_ori)
        print(min_label)