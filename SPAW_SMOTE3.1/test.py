# result=[]
#
# for i in range(10):
#     result.append(i)
#     result[i]=[]
#     print(i)
#     for j in range (100):
#         result[i].append(j)
# List = ['Mon','Mon','Mon','aon']
#
# result = List.count(List[0])
# print(len(List))

# from dataReading.datasetArff import loadData
# import numpy as np
# #
# points, label = loadData("D:\\PycharmObject\\dataset\\test\\2d-3c.arff")
# # print (label)
# unique, counts = np.unique(label, return_counts=True)
# print(unique)
# print(counts)
# class_stats = dict(zip(unique, counts))
# print(class_stats)

# print(len(points[0]))

# a=[[3,8,1],[1,2,6],[1,5,9]]
# #a.sort(key=lambda x: x[0], reverse=False)
# num = [1,2,3,4,3,2,2,6,6,3,5,5,4,4,6]
# t = list(enumerate(num))
# print(t)

# from sklearn.datasets import make_classification, make_circles
# from matplotlib import pyplot as plt
# from dataPaint import show_dataset
# nb_samples = 500
# X, Y = make_classification(n_samples=nb_samples, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,weights=3)
# #X, Y = make_circles(n_samples=nb_samples, noise=0.1)
# show_dataset(X, Y)
import numpy as np
from sklearn.datasets import make_classification,make_moons
import matplotlib.pyplot as plt
from collections import Counter

# 1.创建类别不平衡的数据集：
# 使用make_classification生成样本数据
# X, y = make_classification(n_samples=5000,
#                            n_features=2,  # 特征个数 = n_informative() + n_redundant + n_repeated
#                            n_informative=2,  # 多信息特征的个数
#                            n_redundant=0,  # 冗余信息，informative特征的随机线性组合
#                            n_repeated=0,  # 重复信息，随机提取n_informative和n_redundant 特征
#                            n_classes=2,  # 分类类别
#                            n_clusters_per_class=1,  # 某一个类别是有几个cluster构成的
#                            weights=[0.01, 0.99],  # 列表类型，权重比
#                            #random_state=0
#                            )
# X,y = make_moons([50, 4950], noise=0.1)
#
# # 2.查看各个标签的样本：
# counter = Counter(y)
# print(counter) # Counter({2: 4674, 1: 262, 0: 64})
#
# # 3.数据集可视化：
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

# from sklearn.cluster import DBSCAN, KMeans
# import numpy as np
# X = np.array([[1, 2], [2, 2], [2, 3],
#              [2, 2], [1, 5], [2, 4]])
# #clustering = DBSCAN(eps=3, min_samples=2).fit(X)
# # print(clustering.labels_)
# # clus_num = len(set(clustering.labels_))-1
# # print(clus_num)
# kmeans = KMeans(n_clusters=1).fit(X)
# print(kmeans.cluster_centers_)
# X = []
# print(type(X))
#
# if isinstance(X, list):       # 判断是否为字符串类型
#     print("It's str.")

# import numpy as np
# import matplotlib.pyplot as plt
#
# fig, ax = plt.subplots(1, 1, figsize=(16, 9))
#
# ax.scatter(1.5, 1, marker='o', color='g')
# plt.show()




import numpy as np
# p2 = np.array([1, 2])
# p1 = np.array([3, 4])
# p3 = np.array([5, 6])
# v = p2 - p1
# print(v)
# m = (p1 + p2) / 2
# print(m)
# # dot()返回的是两个数组的点积
# a = np.dot(v, m)
#
# indices_a = [i for i in indices if np.dot(points[i], v) - a > 0]
# indices_b = [i for i in indices if np.dot(points[i], v) - a < 0]


# from dataset.dataProcessing import data_read_keel,data_write_xls
# filepath = 'D:\\PycharmObject\\annoy\\dataset\\KEEL\\wisconsin.dat'
# data, tag = data_read_keel(filepath)
# data_write_xls(data, "test.xlsx")
# p = np.array([1,3,2,7,5,4,8,9,20,11,6])
#
# data_order = p.argsort()
# data_order = np.flipud(data_order)
#
#
# print(data_order)

import os
for root, dirs, files in os.walk(r"D:\\PycharmObject\\ZQannoy4.1\\dataset\\KEEL\\high"):
    for file in files:
        #获取文件所属目录
        #print(root)
        #print(dirs)
        #print(files)
        #获取文件路径
        print(os.path.join(root, file))


