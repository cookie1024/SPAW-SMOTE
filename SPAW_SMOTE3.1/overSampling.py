from densityAndCenter import dataExtract
from dataReading.datasetArff import loadData
from treeSpaceParting import space
from treeSpaceParting import partData
from minClassLabel import select_label_statistics
from generateData import generateData
import numpy as np
# from imblearn.over_sampling import SMOTE
# import smote_variants as sv



#filepath = "D:\\PycharmObject\\dataset\\test\\2d-3c.arff"
# filepath = 'D:\\PycharmObject\\annoy\\dataset\\Syn\\syn2.arff'
# data, tag = loadData(filepath)

data, tag = generateData()
dim = 2#数据维数
#（算法核心，需要划分的高密度点）
k = 5
extract_data_num = dataExtract(data, k)

#print(extract_data_num)


#空间划分
space_point = space(data, tag, extract_data_num)
#把数据和标签分离，方便后面处理
data,label,leaves,layer = partData(space_point, dim)
#print(space_point)

num_layer = 7  # 在第几层进行smote
print(leaves)
leaves.append(leaves[-1]+1)
layer.append(0)
#print(leaves)
min_label = select_label_statistics(label) # 少数类标签
print(min_label)
add_smote_resampled = []  # 增加的数据集
label_smote_resampled = []  # 增加的数据集所对应的标签
group = []  # 一组
group_label = []
group_num = leaves[0]
num = 0
i = 0
key = 0

while i != (len(space_point)+1):
    if (leaves[i] == group_num):
        #print(group_num)
        if layer[i] != num_layer:
            i = i + 1
            continue
        group.append(num)
        group[num] = []
        # print(num)
        # print(space_point[i][0:wl])
        for k in range(dim):
            group[num].append(data[i][k])
        group_label.append(label[i])
        #print(label)
        num = num + 1
        i = i + 1
    else:
        if len(group) != 0:
            #print ("zq")
            # smote_p = SMOTE() # 建立SMOTE模型对象
            add_smote_resampled.append(key)
            label_smote_resampled.append(key)
            add_smote_resampled[key] = []
            label_smote_resampled[key] = []
            #print(group)
            print(group_label.count(1)+group_label.count(2))
            key = key + 1

        num = 0
        group = []
        group_label = []
        group_num = group_num + 1
#print(group_num)