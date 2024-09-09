from densityAndCenter import dataExtract
from dataReading.datasetArff import loadData
from treeSpaceParting import space
from treeSpaceParting import partData
from minClassLabel import select_label_statistics
from generateData import generateData, generateDataTest
import numpy as np
from dataReading.datasetXls import writeData, loadDataMid, writeDataTest, writeDataReTest
import pandas as pd
# from imblearn.over_sampling import SMOTE
# import smote_variants as sv



#filepath = "D:\\PycharmObject\\dataset\\test\\2d-3c.arff"
# filepath = 'D:\\PycharmObject\\annoy\\dataset\\Syn\\syn2.arff'
# data, tag = loadData(filepath)

def test(space_point, dim=2):
    #dim: 维数 k:KNN紧邻数

    #把数据和标签分离，方便后面处理
    data, label, leaves, layer = partData(space_point, dim)
    #print(space_point)
    num_layer = 7  # 在第几层进行smote
    #print(leaves)
    # leaves.append(leaves[-1]+1)
    # layer.append(0)
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
    group_layer = layer[0]
    min_layer_count = np.zeros(int(max(set(layer)))+1)
    max_layer_count = np.zeros(int(max(set(layer)))+1)

    while i != (len(space_point)+1):
        if i != len(space_point) and leaves[i] == group_num:
            #print(group_num)
            group.append(num)
            group[num] = []
            # print(num)
            # print(space_point[i][0:wl])
            for k in range(dim):
                group[num].append(data[i][k])
            group_label.append(label[i])
            group_layer = int(layer[i])
            #print(label)
            num = num + 1
            i = i + 1
        else:
            #进入一组数据，是否需要过采样
            if i == len(space_point):
                i = i - 1
                # print ("zq")
                # smote_p = SMOTE() # 建立SMOTE模型对象
                add_smote_resampled.append(key)
                label_smote_resampled.append(key)
                add_smote_resampled[key] = []
                label_smote_resampled[key] = []
                # print(group)
                # print(group_layer)
                min_layer_count[group_layer] = min_layer_count[group_layer] + group_label.count(0)
                max_layer_count[group_layer] = max_layer_count[group_layer] + group_label.count(1)
                # print(group_label.count(1)+group_label.count(2))
                key = key + 1
                i = i + 2
            else:
                #print ("zq")
                # smote_p = SMOTE() # 建立SMOTE模型对象
                add_smote_resampled.append(key)
                label_smote_resampled.append(key)
                add_smote_resampled[key] = []
                label_smote_resampled[key] = []
                #print(group)
                #print(group_layer)
                min_layer_count[group_layer] = min_layer_count[group_layer] + group_label.count(0)
                max_layer_count[group_layer] = max_layer_count[group_layer] + group_label.count(1)
                #print(group_label.count(1)+group_label.count(2))
                key = key + 1

            num = 0
            group = []
            group_label = []
            group_num = group_num + 1

    return min_layer_count, max_layer_count
    #print(min_layer_count)

if __name__ == '__main__':

    #第san次极端测试，Max：Min = 50：1 ---10：1
    sam_Max = 5000
    sam_Min = 50
    k = 10
    ratio = 0.6
    leaf_sizes = 50
    count_result_Min = []
    count_result_Max = []
    #kmean_flag = 1
    for i in range(20):
        data, tag = generateDataTest(sam_Min, sam_Max)
        #（算法核心，需要划分的高密度点）
        extract_data_num = dataExtract(data, k, ratio)
        #print(extract_data_num)
        #空间划分
        space_point = space(leaf_sizes, data, tag, extract_data_num, i)
        #写入数据
        writeDataTest(space_point, i)
        #space_point = loadDataMid('./mid_result/dataProssesing1.xlsx')
        #print(space_point)
        min_data, max_data = test(space_point)
        count_result_Min.append(min_data)
        count_result_Max.append(max_data)
        #count_result.append(test(space_point))
        #writeData(count_result)
        # sam_Min = sam_Min + 20
        # sam_Max = sam_Max - 20
        #print(count)
    writeDataReTest(count_result_Min, 1)
    writeDataReTest(count_result_Max, 2)


    # #调用中间结果
    # count_result_Min = []
    # count_result_Max = []
    # file_name = "D:\\PycharmObject\\annoy\\mid_result\\dataProssesing"
    # #kmean_flag = 1
    # for i in range(30):
    #     #写入数据
    #     name = file_name+str(i)+".xlsx"
    #     space_point = loadDataMid(name)
    #     #space_point = loadDataMid('./mid_result/dataProssesing1.xlsx')
    #     #print(space_point)
    #     min_data, max_data = test(space_point)
    #     count_result_Min.append(min_data)
    #     count_result_Max.append(max_data)
    #     #writeData(count_result)
    #     # sam_Min = sam_Min + 20
    #     # sam_Max = sam_Max - 20
    #     #print(count)
    # writeDataReTest(count_result_Min, 1)
    # writeDataReTest(count_result_Max, 2)
