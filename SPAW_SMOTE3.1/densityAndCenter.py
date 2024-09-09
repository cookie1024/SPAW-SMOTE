import annoy as annoy
import pandas as pd
import numpy as np
from sklearn import preprocessing
from annoy import AnnoyIndex
import matplotlib.pyplot as pyplot
from matplotlib.patches import Circle
from dataReading.datasetArff import loadData


def showCircle(data, radiusKNN):#画图
    fig = pyplot.figure()
    ax = fig.add_subplot(111)

    pyplot.plot(data[1][0], data[1][1], color='red', label='A', marker='o', linestyle='None')
    circle = Circle(xy=(data[1][0], data[1][1]), radius=radiusKNN[1], alpha=0.2, color='red')
    ax.add_patch(circle)
    ax.arrow(data[1][0], data[1][1], radiusKNN[1], data[0][1], length_includes_head=True, head_width=0.04,
             head_length=0.05, fc='red', ec='red')

    pyplot.plot(data[2][0], data[2][1], color='black', label='B', marker='o', linestyle='None')
    circle = Circle(xy=(data[2][0], data[2][1]), radius=radiusKNN[2], alpha=0.2, color='black')
    ax.add_patch(circle)
    ax.arrow(data[2][0], data[2][1], radiusKNN[2], data[0][1], length_includes_head=True, head_width=0.04,
             head_length=0.05, fc='black', ec='black')

    pyplot.plot(data[5][0], data[5][1], color='cyan', label='C', marker='o', linestyle='None')
    circle = Circle(xy=(data[5][0], data[5][1]), radius=radiusKNN[5], alpha=0.2, color='cyan')
    ax.add_patch(circle)
    ax.arrow(data[5][0], data[5][1], radiusKNN[5], data[0][1], length_includes_head=True, head_width=0.04,
             head_length=0.05, fc='cyan', ec='cyan')

    pyplot.plot(data[0][0], data[0][1], color='grey', label='D', marker='o', linestyle='None')
    circle = Circle(xy=(data[0][0], data[0][1]), radius=radiusKNN[0], alpha=0.2, color='grey')
    # pyplot.plot(data[0][0],data[0][1],color='grey',label='A')
    ax.add_patch(circle)
    ax.arrow(data[0][0], data[0][1], radiusKNN[0], data[0][1], length_includes_head=True, head_width=0.04,
             head_length=0.05, fc='grey', ec='grey')

    pyplot.plot(data[3][0], data[3][1], color='green', label='E', marker='o', linestyle='None')
    circle = Circle(xy=(data[3][0], data[3][1]), radius=radiusKNN[3], alpha=0.2, color='green')
    ax.add_patch(circle)
    ax.arrow(data[3][0], data[3][1], radiusKNN[3], data[0][1], length_includes_head=True, head_width=0.04,
             head_length=0.05, fc='green', ec='green')

    pyplot.plot(data[4][0], data[4][1], color='purple', label='F', marker='o', linestyle='None')
    circle = Circle(xy=(data[4][0], data[4][1]), radius=radiusKNN[4], alpha=0.2, color='purple')
    ax.add_patch(circle)
    ax.arrow(data[4][0], data[4][1], radiusKNN[4], data[0][1], length_includes_head=True, head_width=0.04,
             head_length=0.05, fc='purple', ec='purple')

    pyplot.tight_layout()
    pyplot.legend(loc='lower left', fontsize=12)
    pyplot.axis('equal')
    pyplot.savefig(r'circle.svg', bbox_inches='tight', dpi=1200)
    pyplot.show()
    return " "


def searchKNN(dataCircle, neighbourNumber):

    model = AnnoyIndex(dataCircle.shape[1], "euclidean")
    for i in range(dataCircle.shape[0]):
        # a.add_item(i, v) 添加向量元素v到索引树，其中，i应该为非负整数
        model.add_item(i, dataCircle[i])
    model.build(10)#建几棵树

    pointKNN = []
    asnn = []
    # asnn_remove = []
    for i in range(dataCircle.shape[0]):
        # False:    [[0, 2, 3, 1, 4], [1, 2, 0, 3, 4], [2, 1, 0, 3, 4], [3, 4, 0, 2, 1], [4, 3, 0, 2, 1]]
        # True:     [([0, 2, 3, 4, 1], [0.0, 0.4739063084125519, 1.27005934715271, 1.44548499584198, 1.8466510772705078]), ([1, 4, 3, 2, 0], [0.0, 0.6564297080039978, 0.7244844436645508, 1.4114969968795776, 1.8466510772705078]), ([2, 0, 3, 4, 1], [0.0, 0.4739063084125519, 0.934003472328186, 0.9719761610031128, 1.4114969968795776]), ([3, 1, 4, 2, 0], [0.0, 0.7244844436645508, 0.8284012079238892, 0.934003472328186, 1.27005934715271]), ([4, 1, 3, 2, 0], [0.0, 0.6564297080039978, 0.8284012079238892, 0.9719761610031128, 1.44548499584198])]
        asnn.append(model.get_nns_by_item(i, neighbourNumber, search_k=-1, include_distances=True))
        # asnn_remove.append(model.get_nns_by_item(i, neighbourNumber, search_k=-1, include_distances=False))
        pointKNN = np.asarray(asnn)
    return pointKNN


def neighborsWithDistance(nn):
    indptr = np.asarray(nn[:, 0, :][:, 1:], dtype=np.int64)
    distance = nn[:, 1, :][:, 1:]
    return (indptr, distance)


def radiusGet(pointKNN, dataShape):
    #radius = np.zeros(dataShape)
    #print(pointKNN[1][:,0:])
    radius = np.mean(pointKNN[1][:, 0:], axis=1)
    return radius


def neighborsByRadius(KneighborsWithDistance, radius):
    neighborsbyRadius = [np.extract(KneighborsWithDistance[1][i] <= radius[i], KneighborsWithDistance[0][i]) for i in
                         range(radius.shape[0])]
    return neighborsbyRadius


def ponintNeighborCount(neighborsByRadius):
    pointCount = [len(i) for i in neighborsByRadius]
    pointCount = np.asarray(pointCount)
    #print(pointCount)
    #print(-pointCount)
    index = np.argsort(-pointCount) #排序，返回标签
    #print(index)
    pointCount = np.sort(-pointCount)
    #print(pointCount)

    return (-pointCount, index)


def spreadAbility(pointKNN, neighborsByRadius, dataShape):
    spreadnumber = []
    spreadability = []
    temp = []
    # temp.clear()
    # print(neighborsByRadius)
    count = 0
    for i in neighborsByRadius:
        temp.append(count)
        temp = np.append(temp, i)
        for j in i:
            '''
            print("count = ", count)
            print("i = ", i)
            print("temp = ", temp)
            print("neighborsByRadius[j] = ", neighborsByRadius[j])
            '''
            temp = np.append(temp, neighborsByRadius[j])
            # print(neighborsByRadius[j])
            temp = np.ravel(temp)
        # temp = (reduce(operator.add, temp))
        temp = list(np.unique(temp))
        # print(temp)
        spreadnumber.append(len(temp))
        temp.clear()
        count = count + 1
    # print(spreadnumber)
    strongNeighbor = []
    strongNeighborNumber = []
    count = 0
    # print(neighborsByRadius)
    for i in neighborsByRadius:
        spreadability.append(spreadnumber[count])
        # print(pointKNN[1][count:count+1,0:1])
        for j in i:
            spreadability[count] = spreadability[count] + spreadnumber[j]
            strongNeighbor.append(spreadnumber[j])
        maxNeighbor = np.max(strongNeighbor)
        # print(strongNeighbor)
        # print("i = ",i)
        for j in i:
            if spreadnumber[j] == maxNeighbor:
                strongNeighborNumber.append(j)
                # print(j)
        # i = np.asarray(i)
        # indexStrongNeighbor = np.where(spreadnumber[i]== maxNeighbor)
        # print("strongNeighborNumber = ", strongNeighborNumber)
        for k in strongNeighborNumber:
            # print("spreadnumber[k] = ", spreadnumber[k])

            index = np.where(i == k)
            index = np.ravel(index)
            # index =
            # print("index = ", index)
            for m in index:
                # print("pointKNN[1][count:count+1,m:m+1] = ", pointKNN[1][count:count+1,m:m+1])
                # print()
                # log.error(pointKNN[1][count:count+1,m:m+1])
                if pointKNN[1][count:count + 1, m:m + 1] != 0:
                    spreadability[count] = spreadability[count] + spreadnumber[k] / pointKNN[1][count:count + 1,
                                                                                    m:m + 1]
                else:
                    continue
        count = count + 1
        strongNeighbor.clear()
        strongNeighborNumber.clear()
    # print(spreadability)
    return np.ravel(np.asarray(spreadability, dtype=np.int64))


def centerGet(spreadAbility, dataShape, ratio):
    # print(spreadAbility)
    # maxSpreadAbility = np.max(spreadAbility)
    # log.error(maxSpreadAbility)
    # threshold = maxSpreadAbility * 0.1
    spreadAbilitySort = np.sort(-spreadAbility)
    #print(spreadAbilitySort)
    spreadAbilitySort = -spreadAbilitySort
    #print(spreadAbilitySort)
    threshold = spreadAbilitySort[int(dataShape * ratio)]
    # log.error(threshold)
    #print(threshold)
    center = np.zeros(dataShape, dtype=np.int64)
    center[np.where(spreadAbility > threshold)] = 1
    return center


def plotfig(data, sample, name):
    # fig=pyplot.figure(1)
    colors = np.array(['dimgrey', 'red', 'black', 'orange', 'blue', 'pink', 'black', 'red ', 'cyan'])
    # pyplot.subplot(121)
    # pyplot.title('corePoint')
    # pyplot.axis('equal')
    pyplot.scatter(data[:, 0], data[:, 1], marker='o', c=colors[sample])
    pyplot.axis('equal')
    pyplot.savefig(name, bbox_inches='tight', dpi=1200)
    pyplot.show()


def dataExtract(data, K, ratio):
    #data, flag = loadData(filepath)
    #data = np.asarray(data)
    #数据归一化
    data = preprocessing.MinMaxScaler().fit_transform(data)#数据归一化
    dataShape = data.shape[0]#数据个数
    #print(data.shape[0])
    #print(data.shape[1])
    # log.error(data)
    #上面是获取数据

    #获取最近的k-1个点和距离
    pointKNN = searchKNN(data, K)
    #print(pointKNN)
    pointKNN = neighborsWithDistance(pointKNN)
    #print(pointKNN)

    #求距离均值
    radius = radiusGet(pointKNN, dataShape)
    #print(radius)

    #返回所计算的半径内的个数
    neighborsbyradius = neighborsByRadius(pointKNN, radius)
    #print(neighborsbyradius)
    ponintneighborcount = ponintNeighborCount(neighborsbyradius)
    #print(ponintneighborcount)

    #返回密度
    spreadability = spreadAbility(pointKNN, neighborsbyradius, dataShape)
    #print(spreadability)
    center = centerGet(spreadability, dataShape, ratio)
    #print(center)
    return center, spreadability

def data_den_order(data, K=10):
    # data, flag = loadData(filepath)
    data = np.asarray(data)
    # 数据归一化
    data = preprocessing.MinMaxScaler().fit_transform(data)  # 数据归一化
    dataShape = data.shape[0]  # 数据个数
    # print(data.shape[0])
    # print(data.shape[1])
    # log.error(data)
    # 上面是获取数据

    # 获取最近的k-1个点和距离
    pointKNN = searchKNN(data, K)
    # print(pointKNN)
    pointKNN = neighborsWithDistance(pointKNN)
    # print(pointKNN)

    # 求距离均值
    radius = radiusGet(pointKNN, dataShape)
    # print(radius)

    # 返回所计算的半径内的个数
    neighborsbyradius = neighborsByRadius(pointKNN, radius)
    # print(neighborsbyradius)
    ponintneighborcount = ponintNeighborCount(neighborsbyradius)
    # print(ponintneighborcount)

    # 返回密度
    spreadability = spreadAbility(pointKNN, neighborsbyradius, dataShape)
    #密度由大到小进行排列，密度大的分类信息更加多
    den_order = spreadability.argsort()
    den_order = np.flipud(den_order)

    return den_order



if __name__ == '__main__':
    '''
    synLittle K = 4
    synOne K = 20
    '''
    K = 3
    # deep = 10
    #data = pd.read_csv('D://BaiduNetdiskDownload//syn1.csv', sep=',')
    # readName = 'datasets/syn/d.csv'
    # toCenter = 'D://BaiduNetdiskDownload//3Center.csv'
    # plotName = 'D://BaiduNetdiskDownload//Fig4_syn3_10.eps'
    filepath = 'D:\\PycharmObject\\dataset\\test\\2d-3c.arff'
    data, flag = loadData(filepath)

    data = np.asarray(data)
    data = preprocessing.MinMaxScaler().fit_transform(data)#数据归一化
    dataShape = data.shape[0]#数据个数
    #print(data.shape[0])
    # log.error(data)
    #上面是获取数据

    #获取最近的k-1个点和距离
    pointKNN = searchKNN(data, K)
    print(pointKNN)
    pointKNN = neighborsWithDistance(pointKNN)
    #print(pointKNN)

    #求距离均值
    radius = radiusGet(pointKNN, dataShape)
    #print(radius)

    #返回所计算的半径内的个数
    neighborsbyradius = neighborsByRadius(pointKNN, radius)
    #print(neighborsbyradius)
    ponintneighborcount = ponintNeighborCount(neighborsbyradius)
    #print(ponintneighborcount)

    #返回密度
    spreadability = spreadAbility(pointKNN, neighborsbyradius, dataShape)
    #print(spreadability.argsort())
    data_order = spreadability.argsort()
    #print(data_order[0])

    ratio = 0.6
    center = centerGet(spreadability, dataShape, ratio)
    #print(len(center))
    #showCircle(data, radius)

    # centerToCSV = center
    # centerToCSV = pd.DataFrame(center)
    # centerToCSV.to_csv(toCenter, index=False, header=True)
    # plotfig(data, center, plotName)

'''
targetArray = ['blood','breastcancer','breasttissue','ecoli','glass',
                'haberman','heart','iris','kdiabetes','libra',
                'liverdisorders','pima','segment','vehicle','wine']
targetArray = []
for target in targetArray:
    # target = 'wine'
    readName = 'datasets/uci/'+target+'.csv'
    toCenter = 'datasets/uci/'+target+'_Center.csv'
    plotName = 'pictures/uci/'+target+'_Center.eps'

    log.error(readName)
    data = pd.read_csv(readName,sep=',',header="infer")
    # data = data.drop_duplicates()
    if target == 'blood':
        label = data['label']
        data = np.asarray(data)
        data = preprocessing.MinMaxScaler().fit_transform(data[:,0:4])

    # breastcancer
    if target == 'breastcancer':
        label_mapping = {'benign':0,'malignant':1}
        # log.error(data['label'])
        label = data['label'].map(label_mapping)
        # log.error(label)
        # exit(0)
        data = np.asarray(data)
        data = preprocessing.MinMaxScaler().fit_transform(data[:,1:10])

    #  breasttissue
    if target == 'breasttissue':
        label_mapping = {'car':0,'fad':1,'mas':2,'gla':3,'con':4,'adi':5}
        # log.error(data['label'])
        label = data['label'].map(label_mapping)
        # log.error(label)
        # exit(0)
        data = np.asarray(data)
        data = preprocessing.MinMaxScaler().fit_transform(data[:,2:12])

    #  ecoli
    if target == 'ecoli':
        label = data['label']
        data = np.asarray(data)
        data = preprocessing.MinMaxScaler().fit_transform(data[:,0:7])

    #  glass
    if target == 'glass':
        label = data['label']
        data = np.asarray(data)
        data = preprocessing.MinMaxScaler().fit_transform(data[:,0:9])

    #  haberman
    if target == 'haberman':
        label = data['label']
        data = np.asarray(data)
        data = preprocessing.MinMaxScaler().fit_transform(data[:,0:3])

    #  heart
    if target == 'heart':
        label = data['label']
        data = np.asarray(data)
        data = preprocessing.MinMaxScaler().fit_transform(data[:,0:13])

    #  iris
    if target == 'iris':
        label = data['label']
        data = np.asarray(data)
        data = preprocessing.MinMaxScaler().fit_transform(data[:,0:4])

    # kdiabetes
    if target == 'kdiabetes':
        label = data['label']
        data = np.asarray(data)
        data = preprocessing.MinMaxScaler().fit_transform(data[:,0:9])

    # libra
    if target == 'libra':
        label = data['label']
        data = np.asarray(data)
        data = preprocessing.MinMaxScaler().fit_transform(data[:,0:90])

    # liverdisorders
    if target == 'liverdisorders':
        label = data['label']
        data = np.asarray(data)
        data = preprocessing.MinMaxScaler().fit_transform(data[:,0:6])

    # pima
    if target == 'pima':
        label = data['label']
        data = np.asarray(data)
        data = preprocessing.MinMaxScaler().fit_transform(data[:,0:8])

    # segment
    if target == 'segment':
        label = data['label']
        data = np.asarray(data)
        data = preprocessing.MinMaxScaler().fit_transform(data[:,0:19])

    # vehicle
    if target == 'vehicle':
        label = data['label']
        data = np.asarray(data)
        data = preprocessing.MinMaxScaler().fit_transform(data[:,0:18])

    # wine
    if target == 'wine':
        label = data['label']
        data = np.asarray(data)
        data = preprocessing.MinMaxScaler().fit_transform(data[:,0:13])

    dataShape = data.shape[0]
    # log.error(data)
    pointKNN = searchKNN(data,K)
    pointKNN = neighborsWithDistance(pointKNN)
    radius = radiusGet(pointKNN, dataShape)
    neighborsbyradius = neighborsByRadius(pointKNN,radius)
    ponintneighborcount = ponintNeighborCount(neighborsbyradius)
    spreadability = spreadAbility(pointKNN,neighborsbyradius,dataShape)
    center = centerGet(spreadability,dataShape)
    # centerToCSV = center
    # centerToCSV = pd.DataFrame(center)
    # centerToCSV.to_csv(toCenter,index=False,header = True)
    plotfig(data,center,plotName)
'''