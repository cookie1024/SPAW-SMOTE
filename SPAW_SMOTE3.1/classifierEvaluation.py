import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from generateData import generateData
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import pandas as pd
from dataset.dataProcessing import data_read_keel_np
import math
import smote_variants as sv

def evaluation_kfold(data, label):
    #获取数据
    #iris = datasets.load_iris()
    #print(iris.data)
    #数据标准化
    # iris_trans = preprocessing.MinMaxScaler().fit_transform(iris.data)
    # print(iris_trans)
    #设置分类器
    clf = svm.SVC(kernel='rbf')#‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    #clf = tree.DecisionTreeClassifier()
    #clf = RandomForestClassifier(n_estimators=10)
    #是否封装标准化
    #clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))

    #k-折交叉验证
    #参数cv 几折交叉
    #scoring 不同的评价方法修改
    #scores = cross_val_score(clf, data, label, cv=5, scoring='recall')
    #‘accuracy’
    #‘f1’
    #‘recall’
    #‘roc_auc’
    scores_accuracy = cross_val_score(clf, data, label, cv=5, scoring='accuracy')

    scores_recall = cross_val_score(clf, data, label, cv=5, scoring='recall')
    scores_f1 = cross_val_score(clf, data, label, cv=5, scoring='f1')
    scores_roc_auc = cross_val_score(clf, data, label, cv=5, scoring='roc_auc')
    print("accuracy:{0}  recall:{1}  scores_f1:{2}  scores_roc_auc:{3}".format(scores_accuracy.mean(), scores_recall.mean(), scores_f1.mean(), scores_roc_auc.mean()))

    #print(iris.data, iris.target)


def evaluation_kfold_recall(data, label):
    # 获取数据
    # iris = datasets.load_iris()
    # print(iris.data)
    # 数据标准化
    # iris_trans = preprocessing.MinMaxScaler().fit_transform(iris.data)
    # print(iris_trans)
    # 设置分类器
    clf = svm.SVC(kernel='rbf')  # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    # clf = tree.DecisionTreeClassifier()
    # clf = RandomForestClassifier(n_estimators=10)
    # 是否封装标准化
    # clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))

    # k-折交叉验证
    # 参数cv 几折交叉
    # scoring 不同的评价方法修改
    # scores = cross_val_score(clf, data, label, cv=5, scoring='recall')
    # ‘accuracy’
    # ‘f1’
    # ‘recall’
    # ‘roc_auc’
    scores_recall = cross_val_score(clf, data, label, cv=5, scoring='recall')

    return scores_recall.mean()

def evaluation_kfold_roc_auc(data, label):
    # 获取数据
    # iris = datasets.load_iris()
    # print(iris.data)
    # 数据标准化
    # iris_trans = preprocessing.MinMaxScaler().fit_transform(iris.data)
    # print(iris_trans)
    # 设置分类器
    clf = svm.SVC(kernel='rbf')  # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    # clf = LogisticRegression()
    # clf = RandomForestClassifier(n_estimators=10)
    # 是否封装标准化
    # clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))

    # k-折交叉验证
    # 参数cv 几折交叉
    # scoring 不同的评价方法修改
    # scores = cross_val_score(clf, data, label, cv=5, scoring='recall')
    # ‘accuracy’
    # ‘f1’
    # ‘recall’
    # ‘roc_auc’
    scores_roc_auc = cross_val_score(clf, data, label, cv=5, scoring='roc_auc')
    #print(scores_roc_auc)
    return scores_roc_auc.mean()




def evaluation_kfold_f1(data, label):
    # 获取数据
    # iris = datasets.load_iris()
    # print(iris.data)
    # 数据标准化
    # iris_trans = preprocessing.MinMaxScaler().fit_transform(iris.data)
    # print(iris_trans)
    # 设置分类器
    # clf = svm.SVC(kernel='rbf')  # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    # clf = LogisticRegression()
    clf = RandomForestClassifier(n_estimators=10)
    # 是否封装标准化
    # clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))

    # k-折交叉验证
    # 参数cv 几折交叉
    # scoring 不同的评价方法修改
    # scores = cross_val_score(clf, data, label, cv=5, scoring='recall')
    # ‘accuracy’
    # ‘f1’
    # ‘recall’
    # ‘roc_auc’

    scores_f1 = cross_val_score(clf, data, label, cv=5, scoring='f1')
    return scores_f1.mean()
    #print("accuracy:{0}  recall:{1}  scores_f1:{2}  scores_roc_auc:{3}".format(scores_accuracy.mean()))

    # print(iris.data, iris.target)




def evaluation_kfold_Gmean(data, label):
    if type(data) is not np.ndarray:
        data = np.array(data)
    if type(label) is not np.ndarray:
        label = np.array(label)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)  # 初始化KFold
    G_mean = []
    gnum = 0
    for train_index, test_index in kf.split(data):  # 调用split方法切分数据
        # 建立模型，并对训练集进行测试，求出预测得分
        # 划分训练集和测试集
        X_train, X_test = data[train_index], data[test_index]
        Y_train, Y_test = label[train_index], label[test_index]
        # clf = svm.SVC(kernel='rbf')
        # clf = LogisticRegression()
        clf = RandomForestClassifier(n_estimators=10)

        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)
        # value_matrix = confusion_matrix(Y_test, y_pred)

        tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
        G_mean.append(math.sqrt(tp/(tp+fn)*tn/(fp+tn)))

    g_mean_np = np.array(G_mean)
    #print(g_mean_np)
    #print(g_mean_np.mean())

    return g_mean_np.mean()


def evaluation_kfold_auc_f1_gmean(data, label):
    if type(data) is not np.ndarray:
        data = np.array(data)
    if type(label) is not np.ndarray:
        label = np.array(label)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)  # 初始化KFold
    auc = []
    f_measure = []
    G_mean = []
    for train_index, test_index in kf.split(data):  # 调用split方法切分数据
        # 建立模型，并对训练集进行测试，求出预测得分
        # 划分训练集和测试集
        X_train, X_test = data[train_index], data[test_index]
        Y_train, Y_test = label[train_index], label[test_index]
        # clf = svm.SVC(kernel='rbf', probability=True)
        clf = LogisticRegression()
        # clf = RandomForestClassifier(n_estimators=20)

        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)
        # value_matrix = confusion_matrix(Y_test, y_pred)

        tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
        G_mean.append(math.sqrt(tp/(tp+fn)*tn/(fp+tn)))
        f_measure.append(2*tp/(2*tp+fp+fn))
        auc.append(roc_auc_score(Y_test, clf.predict_proba(X_test)[:, 1]))



    f_measure_np = np.array(f_measure)
    g_mean_np = np.array(G_mean)
    auc_np = np.array(auc)
    #print(g_mean_np)
    #print(g_mean_np.mean())

    return auc_np.mean(), f_measure_np.mean(), g_mean_np.mean()


def evaluation_division(data, label):
    #划分数据
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.4, random_state=None)
    clf = svm.SVC(kernel='poly', C=1).fit(X_train, y_train)
    print(clf.score(X_test, y_test))


def write_evaluation(list, filename):
    # 二维list
    n = len(list[0])
    s = np.arange(n)
    s.tolist()
    # list转dataframe
    df = pd.DataFrame(list, columns=s)
    # 保存到本地excel
    df.to_excel(filename, index=False)


if __name__ == '__main__':

    # data, target = generateData()
    filepath = "D:\\PycharmObject\\ZQannoy5.2\\dataset\\KEEL\\normal\\yeast1.dat"
    data_ori, tag_ori, data_ori_np, tag_ori_np = data_read_keel_np(filepath)
    smote_only = sv.SMOTE()
    x_smote_resampled, y_smote_resampled = smote_only.fit_resample(data_ori_np, tag_ori_np)
    print(evaluation_kfold_auc_f1_gmean(x_smote_resampled, y_smote_resampled))







