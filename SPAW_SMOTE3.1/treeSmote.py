import numpy as np
from annoy import AnnoyIndex
from dataReading.datasetArff import loadData
import random
from minClassLabel import select_label_statistics
from dataPaint import show_clustered_dataset, show_dataset
from minClassLabel import select_label_statistics, select_two_label_statistics
from treeSpacePartingDim import partData, data_str_to_flow

def data_label_judgment(label, data, min_label):
    num = 0
    selct_num = []
    flag = 0
    # min_labels, maj_labels = select_label_statistics(label)
    # min_label = min_labels[0]
    # maj_label = maj_labels[0]
    #k = 1
    for i in range(len(data)):
        if label[i] == min_label:
            num = num + 1
            selct_num.append(i)
    if num > 1:
        flag = 1

    # if k > maj_min_ratio_int:
    #     k = maj_min_ratio_int

    return flag, selct_num

def search_ANNOY(data, neighbourNumber = 4):
    #neighbourNumber = 4
    point_num = len(data)
    # 建树
    model = AnnoyIndex(data.shape[1], "euclidean")
    for i in range(data.shape[0]):
        # a.add_item(i, v) 添加向量元素v到索引树，其中，i应该为非负整数
        model.add_item(i, data[i])
    model.build(1)  # 建几棵树
    pointKNN = []
    asnn = []
    for i in range(data.shape[0]):
        # False:    [[0, 2, 3, 1, 4], [1, 2, 0, 3, 4], [2, 1, 0, 3, 4], [3, 4, 0, 2, 1], [4, 3, 0, 2, 1]]
        # True:     [([0, 2, 3, 4, 1], [0.0, 0.4739063084125519, 1.27005934715271, 1.44548499584198, 1.8466510772705078]), ([1, 4, 3, 2, 0], [0.0, 0.6564297080039978, 0.7244844436645508, 1.4114969968795776, 1.8466510772705078]), ([2, 0, 3, 4, 1], [0.0, 0.4739063084125519, 0.934003472328186, 0.9719761610031128, 1.4114969968795776]), ([3, 1, 4, 2, 0], [0.0, 0.7244844436645508, 0.8284012079238892, 0.934003472328186, 1.27005934715271]), ([4, 1, 3, 2, 0], [0.0, 0.6564297080039978, 0.8284012079238892, 0.9719761610031128, 1.44548499584198])]
        asnn.append(model.get_nns_by_item(i, neighbourNumber, search_k=-1, include_distances=False))
        pointKNN = np.asarray(asnn)
    #print(pointKNN)
    return pointKNN




def region_smote(data, label, dim, min_label, region_add):
    add_point = []
    add_label = []
    add_num = -1
    # remove_data = []  # 减少的数据集
    # remove_label = []  # 减少的数据集对应的标签

    if len(data) < 2:
        return add_point, add_label


    #data_KNN = search_ANNOY(data)
    #smote_num = []
    # print(label[0])
    # print(min_label)
    # min_label = min_labels[0]
    #print(min_label)
    #print(data_KNN)
    flag, select_num = data_label_judgment(label, data, min_label)
    if flag == 0:
        return add_point, add_label
    #print(region_add)
    new_point_num = len(label)

    for i in range(round(region_add)):
        add_num = add_num + 1
        add_point.append(add_num)
        add_point[add_num] = []
        data.append(new_point_num)
        data[new_point_num] = []
        # point_nums = random.sample(select_num, 2)
        # point_num1 = point_nums[0]
        # point_num2 = point_nums[1]
        for k in range(dim):
            point_nums = random.sample(select_num, 2)
            point_num1 = point_nums[0]
            point_num2 = point_nums[1]
            #print(point_num)
            dis = abs(data[point_num1][k]-data[point_num2][k])
            # print(data[i][k])
            # print(data[point_num][k])
            # print(dis)
            ratio = random.random()
            if data[point_num1][k] < data[point_num2][k]:
                s_dis = data[point_num1][k] + dis * ratio
            else:
                s_dis = data[point_num2][k] + dis * ratio
            add_point[add_num].append(s_dis)
            data[new_point_num].append(s_dis)


        add_label.append(min_label)
        # print(new_point_num)
        select_num.append(new_point_num)

        new_point_num = new_point_num + 1



    return add_point, add_label

# def region_smote(data, label, dim, min_label, region_add):
#     add_point = []
#     add_label = []
#     add_num = -1
#     # remove_data = []  # 减少的数据集
#     # remove_label = []  # 减少的数据集对应的标签
#
#     if len(data) < 2:
#         return add_point, add_label
#
#
#     #data_KNN = search_ANNOY(data)
#     #smote_num = []
#     # print(label[0])
#     # print(min_label)
#     # min_label = min_labels[0]
#     #print(min_label)
#     #print(data_KNN)
#     flag, select_num = data_label_judgment(label, data, min_label)
#     if flag == 0:
#         return add_point, add_label
#     #print(region_add)
#     new_point_num = len(label)
#
#     for i in range(int(region_add)):
#         add_num = add_num + 1
#         add_point.append(add_num)
#         add_point[add_num] = []
#         data.append(new_point_num)
#         data[new_point_num] = []
#
#         point_nums = random.sample(select_num, 2)
#         point_num1 = point_nums[0]
#         point_num2 = point_nums[1]
#         point_center = []
#         dis_k = 0
#         for k in range(dim):
#             point_center.append((data[point_num1][k] + data[point_num2][k]) / 2)
#             dis_k = dis_k + (data[point_num1][k] - data[point_num2][k]) ** 2
#             #print(point_num)
#             # print(data[i][k])
#             # print(data[point_num][k])
#             # print(dis)
#         dis_r = (dis_k ** 0.5)/2
#         for k in range(dim):
#             ratio = -1 + 2 * random.random()
#             s_dis = point_center[k] + dis_r * ratio
#             add_point[add_num].append(s_dis)
#             data[new_point_num].append(s_dis)
#         add_label.append(min_label)
#         select_num.append(new_point_num)
#
#         new_point_num = new_point_num + 1
#
#
#
#     return add_point, add_label



def test(space_point, dim, proportion, data_ori_np, tag_ori):
    #dim: 维数 k:KNN紧邻数

    #把数据和标签分离，方便后面处理
    data, label, leaves, layer = partData(space_point, dim)
    layer_max = max(set(layer))
    zq_smote_num_layer = layer_max+1 # 在第几层进行smote

    #print(leaves)
    # leaves.append(leaves[-1]+1)
    # layer.append(0)
    #print(leaves)
    min_label, maj_label = select_two_label_statistics(label) # 少数类标签
    print("少数类标签:{}".format(min_label))


    add_smote_resampled = []  # 增加的数据集
    label_smote_resampled = []  # 增加的数据集所对应的标签
    remove_smote_resampled = [] #减少的数据集
    remove_label_resampled = [] #减少的数据集对应的标签
    remove_sample_num = [] #减少的数据对应的原始序号
    group = []  # 一组
    group_label = []
    group_num = leaves[0]
    num = 0
    i = 0
    #key = 0
    group_layer = layer[0]
    #proportion_group = proportion[int(group_num)]
    # min_layer_count = np.zeros(int(max(set(layer)))+1)
    # max_layer_count = np.zeros(int(max(set(layer)))+1)

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
            # print(group_num)
            proportion_group = proportion[int(group_num)]
            #print(proportion_group)
            #print(group_label)
            if i == len(space_point):
                i = i - 1
                # print ("zq")
                # smote_p = SMOTE() # 建立SMOTE模型对象
                if group_layer < zq_smote_num_layer:
                    add_1, add_2 = region_smote(group, group_label, dim, min_label, proportion_group)
                    add_length = len(add_2)
                    if add_length > 0:
                        for pt in range(add_length):
                            add_smote_resampled.append(add_1[pt])
                            label_smote_resampled.append(add_2[pt])
                #print(group)
                #print(group_num)
                # min_layer_count[group_layer] = min_layer_count[group_layer] + group_label.count(0)
                # max_layer_count[group_layer] = max_layer_count[group_layer] + group_label.count(1)
                # print(group_label.count(1)+group_label.count(2))
                i = i + 2
            else:

                #print ("zq")
                # smote_p = SMOTE() # 建立SMOTE模型对象
                if group_layer < zq_smote_num_layer:
                    add_1, add_2 = region_smote(group, group_label, dim, min_label, proportion_group)
                    add_length = len(add_2)
                    # print(len(add_1))
                    # print(len(add_2))
                    if add_length > 0:
                        for pt in range(add_length):
                            add_smote_resampled.append(add_1[pt])
                            label_smote_resampled.append(add_2[pt])
                #print(group)
                #print(group_num)
                # min_layer_count[group_layer] = min_layer_count[group_layer] + group_label.count(0)
                # max_layer_count[group_layer] = max_layer_count[group_layer] + group_label.count(1)
                #print(group_label.count(1)+group_label.count(2))
                #key = key + 1
            #print(add_smote_resampled)
            num = 0
            group = []
            group_label = []
            group_num = group_num + 1
    #print(type(add_smote_resampled))
    add_smote_resampled_np = np.array(add_smote_resampled)
    #print(add_smote_resampled_np)
    if len(add_smote_resampled_np) == 0:
        data_mid = data_ori_np
        label_mid = tag_ori
    else:
        data_mid = np.concatenate((data_ori_np, add_smote_resampled_np), axis=0)
        label_mid = tag_ori + label_smote_resampled
    print("增加了{0}个少数类样本".format(len(label_smote_resampled)))
    return data_mid, label_mid
    #print(min_layer_count)









if __name__ == '__main__':
    filepath = 'D:\\PycharmObject\\annoy\\dataset\\Syn\\syn2.arff'
    data, tag = loadData(filepath)
    # show_dataset(data, tag)
    print(data)
    p1, p2 = region_smote(data, tag, 2)
    print(p1)
    #p1 = list(map(list, zip(*p1)))
    p1 = np.array(p1)
    show_clustered_dataset(data,tag, p1, p2)

