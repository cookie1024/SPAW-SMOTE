from sklearn.datasets._samples_generator import make_blobs, make_classification, make_moons
from matplotlib import pyplot as plt
from pandas import DataFrame
# generate 2d classification dataset
from sklearn.cluster import KMeans

def generateData():
    X, y = make_blobs(n_samples=[5000, 50],
                           n_features=2, centers=None, cluster_std=2, random_state=2021)

    return X, y

def generateDataTest(sample_Min, sample_Max):
    X, y = make_blobs(n_samples=[sample_Min, sample_Max],
                      n_features=2, centers=None, cluster_std=1)

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
    #X, y = make_moons([50, 5000], noise=0.1)

    return X,y

if __name__ == '__main__':
    X, y = generateDataTest(2000, 50)
    estimator = KMeans(n_clusters=2)
    estimator.fit(X)
    #label_pred = estimator.labels_  # 获取聚类标签
    centroids = estimator.cluster_centers_  # 获取聚类中心
    print(centroids)

    #print(len(set(y)))
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.scatter(centroids[0][0], centroids[0][1], c='orange',marker='x')
    plt.scatter(centroids[1][0], centroids[1][1], c='orange',marker='x')
    plt.show()