import numpy as np
import matplotlib.pyplot as plt




def show_two_clustered_dataset(data,label, add_data, add_label):
    fig, ax = plt.subplots(1, 1, figsize=(30, 25))
    colorSelect = ['b', 'g', 'r', 'c', 'm', 'y', 'k','orange','deeppink', 'purple']
    shapeSelect = ['o', 'v', 's', 'p', '*', '+', 'D', 'x', 'h', '1']
    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if type(label) is np.ndarray:
        label.tolist()
    label_num = set(label)
    num = len(label_num)
    data_1_x = []
    data_1_y = []
    data_2_x = []
    data_2_y = []
    # for i in range(len(data)):
    #     for j in label_num:
    #        # if label[i] == j:






    plt.show()


def show_dataset(X, Y):
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    colorSelect = ['b', 'g', 'r', 'c', 'y', 'k']
    shapeSelect = ['o', 'v', 's', 'p', '*', '+', 'D', 'x', 'h', '1']

    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if type(Y) is np.ndarray:
        #print(Y)
        Y.tolist()
    Y_num = set(Y)
    print(Y_num)
    #num = len(Y_num)
    for i in range(len(X)):
        for j in Y_num:
            if Y[i] == j:
                # X1 = np.float(X[i][0])
                # X2 = np.float(X[i][1])
                #print(j)
                # print(type(np.float(X[i][0])))
                #print(type(X2))
                cj = int(j)
                ax.scatter(X[i, 0], X[i, 1], marker='o', color=colorSelect[cj])
    plt.show()


def show_clustered_dataset(X1, Y1, X2, Y2):
    fig, ax = plt.subplots(1, 1, figsize=(30, 25))
    colorSelect = ['b', 'g', 'r', 'c', 'm', 'y', 'k','orange','deeppink', 'purple']
    shapeSelect = ['o', 'v', 's', 'p', '*', '+', 'D', 'x', 'h', '1']
    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if type(Y1) is np.ndarray:
        Y1.tolist()
    Y_num = set(Y1)
    #num = len(Y_num)

    for i in range(len(X1)):
        for j in Y_num:
            if Y1[i] == j:
                cj = int(j)
                ax.scatter(X1[i, 0], X1[i, 1], marker='o', color=colorSelect[cj])

    if type(Y2) is np.ndarray:
        Y2.tolist()
    Y_num = set(Y2)
    #num = len(Y_num)
    #画增加的点图
    #求差集，在X2中但不在X1中
    #retD = list(set(X2).difference(set(X1)))
    for i in range(len(X2)):
        for j in Y_num:
            if Y2[i] == j:
                cj = int(j)
                ax.scatter(X2[i, 0], X2[i, 1], marker='*', color=colorSelect[cj])

    plt.show()







# 'b'          蓝色
# 'g'          绿色
# 'r'          红色
# 'c'          青色
# 'm'          品红
# 'y'          黄色
# 'k'          黑色
# 'w'          白色

# ‘.’：点(point marker)
# ‘,’：像素点(pixel marker)
# ‘o’：圆形(circle marker)
# ‘v’：朝下三角形(triangle_down marker)
# ‘^’：朝上三角形(triangle_up marker)
# ‘<‘：朝左三角形(triangle_left marker)
# ‘>’：朝右三角形(triangle_right marker)
# ‘1’：(tri_down marker)
# ‘2’：(tri_up marker)
# ‘3’：(tri_left marker)
# ‘4’：(tri_right marker)
# ‘s’：正方形(square marker)
# ‘p’：五边星(pentagon marker)
# ‘*’：星型(star marker)
# ‘h’：1号六角形(hexagon1 marker)
# ‘H’：2号六角形(hexagon2 marker)
# ‘+’：+号标记(plus marker)
# ‘x’：x号标记(x marker)
# ‘D’：菱形(diamond marker)
# ‘d’：小型菱形(thin_diamond marker)
# ‘|’：垂直线形(vline marker)
# ‘_’：水平线形(hline marker)