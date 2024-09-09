from dataset.dataProcessing import data_read_keel
from treeSpacePartingDim import partData, data_str_to_flow
from densityAndCenter import dataExtract
import numpy as np
from sklearn import preprocessing
from annoy import AnnoyIndex

def is_noise(label,min_label,maj_label):
    label_set = set(label)
    min_num = 0
    maj_num = 0
    if len(label) == 1:
        return False
    else:
        for i in range(len(label)):
            if label[i] == min_label:
                min_num = min_num + 1
            else:
                maj_num = maj_num + 1
    if maj_num/min_num > 5:
        return True


def searchKNN_index(dataCircle, neighbourNumber):
    neighbourNumber = neighbourNumber + 1
    model = AnnoyIndex(dataCircle.shape[1], "euclidean")
    for i in range(dataCircle.shape[0]):
        # a.add_item(i, v) 添加向量元素v到索引树，其中，i应该为非负整数
        model.add_item(i, dataCircle[i])
    model.build(10)#建几棵树

    asnn_remove = []
    for i in range(dataCircle.shape[0]):
        # False:    [[0, 2, 3, 1, 4], [1, 2, 0, 3, 4], [2, 1, 0, 3, 4], [3, 4, 0, 2, 1], [4, 3, 0, 2, 1]]
        # True:     [([0, 2, 3, 4, 1], [0.0, 0.4739063084125519, 1.27005934715271, 1.44548499584198, 1.8466510772705078]), ([1, 4, 3, 2, 0], [0.0, 0.6564297080039978, 0.7244844436645508, 1.4114969968795776, 1.8466510772705078]), ([2, 0, 3, 4, 1], [0.0, 0.4739063084125519, 0.934003472328186, 0.9719761610031128, 1.4114969968795776]), ([3, 1, 4, 2, 0], [0.0, 0.7244844436645508, 0.8284012079238892, 0.934003472328186, 1.27005934715271]), ([4, 1, 3, 2, 0], [0.0, 0.6564297080039978, 0.8284012079238892, 0.9719761610031128, 1.44548499584198])]
        p = model.get_nns_by_item(i, neighbourNumber, search_k=-1, include_distances=False)
        del p[0]
        asnn_remove.append(p)
    return asnn_remove


def re_noise(data, label, pointKNN_remove, re_k, re_w):
    num = 0
    re_noise_data = []
    re_noise_label = []
    min_num = 0
    maj_num = 0
    re_min = 0
    p1 = len(label)
    noise_record = np.zeros(p1)
    for i in range(len(label)):
        if label[i] == 0:
            for j in range(re_k):
                if label[pointKNN_remove[i][j]] == 0:
                    num = num + 1
            if num > re_w:
                re_noise_data.append(data[i])
                re_noise_label.append(label[i])
                maj_num = maj_num + 1


        # else:
        #     re_noise_data.append(data[i])
        #     re_noise_label.append(label[i])
        #     min_num = min_num + 1




        else:
            for j in range(re_k):
                if label[pointKNN_remove[i][j]] == 1:
                    num = num + 1
            if num > re_w:
                re_noise_data.append(data[i])
                re_noise_label.append(label[i])
                min_num = min_num + 1
            else:
                re_min = re_min + 1
        num = 0
    p2 = len(re_noise_label)
    print("过滤了{0}个噪声({1}),剩下{2}个数据,少数类：{3} 多数类：{4}".format((p1-p2), re_min, p2, min_num, maj_num))

    return re_noise_data, re_noise_label, min_num



if __name__ == '__main__':
    k = 10
    ratio = 0.6
    filepath = 'D:\\PycharmObject\\ZQannoy3.6\\dataset\\KEEL\\glass1.dat'
    data, tag = data_read_keel(filepath)
    print(len(data))
    data = data_str_to_flow(data)
    p = searchKNN_index(np.array(data), 5)
    print(p)
    # extract_data_num, pointKNN_remove = dataExtract(data, k, ratio)
    # re_noise_data, re_noise_label = re_noise(data, tag, pointKNN_remove)
    # print(len(re_noise_data))

