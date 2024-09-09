import shapely.geometry as sg
import numpy as np
import random
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb
from sklearn.datasets import make_blobs
import pygraphviz
from dataReading.datasetArff import loadData
from densityAndCenter import dataExtract


def split_points(extract_data_num, layer, dim, ax, graph, poly, points, indices, lw=3.0, lo=0.0, hi=5.0 / 6.0,
                 visitor=None, max_splits=99999, draw_splits=True, splits=None, seed='', leaf_size=1,
                 parent_node_id=None):
    global result
    global tags
    global keys
    global num
    # indices 每个叶子节点的个数
    # join()方法用于将序列中的元素以指定的字符连接生成一个新的字符串。
    # print(poly)
    indices_str = ','.join([str(i) for i in indices])
    # print(indices_str)
    # 生成随机数
    random.seed(indices_str)
    # seed=‘’
    # 获取字符串的哈希值，目的是每个id不一样
    node_id = hash(indices_str + seed)
    #判断是否还有超过两个的高密度点
    point_selection = []
    for i in indices:
        if extract_data_num[i] == 1:
            point_selection.append(i)
    # leaf=true or false
    leaf = (len(indices) <= leaf_size or max_splits == 0 or len(point_selection) < 2)

    # print(indices)
    # 不同种类的vister
    visitor.draw_node(graph, node_id, leaf, indices, lo, hi, splits)
    if parent_node_id:  # 这个parent_node_id 怎么发挥作用的？？？？？？
        # print(parent_node_id)
        # 父节点的话增加边
        graph.add_edge(parent_node_id, node_id)
        layer = layer + 1

    if leaf:
        x = [points[i][0] for i in indices]
        y = [points[i][1] for i in indices]
        ##################################################
        for i in indices:
            result.append(num)
            result[num] = []
            for j in range(dim):
                result[num].append(points[i][j])
            result[num].append(tags[i])
            result[num].append(keys)
            result[num].append(layer)
            # print(result[num])
            num = num + 1
        # 在这里将叶子节点存储############################

        ##lo和hi的作用是什么？
        c1 = hsv_to_rgb((lo + hi) / 2, 1, 1)
        c2 = hsv_to_rgb(random.random() * 5.0 / 6.0, 0.7 + random.random() * 0.3, 0.7 + random.random() * 0.3)
        poly_vor = None

        visitor.visit(ax, poly, poly_vor, c1, c2, x, y, splits)
        keys = keys + 1
        # print(keys)
        return
    ###########################################

    random.seed(','.join([str(i) for i in indices]) + seed)
    # p1，p2是随机找到的两个点
    # p1, p2 = [points[i] for i in random.sample(indices, 2)]
    p11 = [points[i] for i in random.sample(point_selection, 1)]

    p_distance_max = 0
    p_distance = 0
    for i in point_selection:
        for j in range(2):
            p_distance = p_distance + (points[i][j]-p11[0][j])**2
        if p_distance > p_distance_max:
            p_distance_max = p_distance
            p2 = points[i]
        p_distance = 0
    p1 = p11[0]
    # print(p1)
    # print(p2)
    v = p2 - p1
    m = (p1 + p2) / 2
    # dot()返回的是两个数组的点积
    a = np.dot(v, m)

    v_perp = np.array((v[1], -v[0]))
    # 下面没看懂polygon的用法
    big = 1e6
    halfplane_a = sg.Polygon(np.array([m + v_perp * big, m + v * big, m - v_perp * big])).intersection(poly)
    halfplane_b = sg.Polygon(np.array([m + v_perp * big, m - v * big, m - v_perp * big])).intersection(poly)
    # print(halfplane_a)

    if draw_splits:
        # print("")
        ax.add_patch(PolygonPatch(halfplane_a, fc='none', lw=lw, zorder=1))
        ax.add_patch(PolygonPatch(halfplane_b, fc='none', lw=lw, zorder=1))

    if max_splits == 1:
        # print("ceshi")
        # 制作灰线，黑线是在哪里画的？？？？？？
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='gray', lw=2.0, zorder=2)

    # 把所在区域的点进行二分类
    indices_a = [i for i in indices if np.dot(points[i], v) - a > 0]
    indices_b = [i for i in indices if np.dot(points[i], v) - a < 0]

    split_points(extract_data_num, layer, dim, ax, graph, halfplane_a, points, indices_a, lw * 0.8, lo, (lo + hi) / 2,
                 visitor, max_splits - 1, draw_splits, (splits, v, a), seed, leaf_size, node_id)
    split_points(extract_data_num, layer, dim, ax, graph, halfplane_b, points, indices_b, lw * 0.8, (lo + hi) / 2, hi,
                 visitor, max_splits - 1, draw_splits, (splits, -v, -a), seed, leaf_size, node_id)


def draw_poly(ax, poly, c, **kwargs):
    if poly.geom_type == 'Polygon':
        polys = [poly]
    else:
        polys = poly.geoms

    for poly in polys:
        # print('')
        # 添加两个分区的颜色
        ax.add_patch(PolygonPatch(poly, fc=c, zorder=0, **kwargs))


def scatter(ax, x, y):
    # 做出散点图
    ax.scatter(x, y, marker='o', zorder=99, c='black', s=10.0)


class Visitor(object):
    def visit(self, ax, poly, c1, c2, x, y, splits):
        pass

    def node_attrs(self, node_id, leaf, indices, lo, hi):
        label = leaf and len(indices) or ''
        shape = leaf and 'circle' or 'square'
        return dict(label=label, style='filled', fillcolor='%f 1.0 1.0' % ((lo + hi) / 2), fontsize=24, fontname='bold',
                    shape=shape)

    def draw_node(self, graph, node_id, leaf, indices, lo, hi, splits):
        # g.add_node('A')  #建立点
        graph.add_node(node_id, **self.node_attrs(node_id, leaf, indices, lo, hi))


class TreeVisitor(Visitor):
    def visit(self, ax, poly, poly_vor, c1, c2, x, y, splits):
        draw_poly(ax, poly, c1)
        scatter(ax, x, y)


# scatter.png
class ScatterVisitor(Visitor):
    def visit(self, ax, poly, poly_vor, c1, c2, x, y, splits):
        scatter(ax, x, y)


def get_points():
    np.random.seed(0)  # 随机种子
    # make_blobs为了生成聚类的数据集，产生一个数据集和相应的标签
    # X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    # n_samples:表示数据样本点个数,默认值100
    # n_features:表示数据的维度，默认值是2
    # centers: 产生数据的中心点，默认值3
    # cluster_std：数据集的标准差，浮点数或者浮点数序列，默认值1.0
    # center_box：中心确定之后的数据边界，默认值(-10.0, 10.0)
    # shuffle ：洗乱，默认值是True
    # random_state:官网解释是随机生成器的种子
    X, y = make_blobs(500, 2, centers=10, center_box=(-4, 4))
    return X


def space(leaf_sizes, points, tag, extract_data_num, num_pic):
    global result
    global tags
    global keys
    global num
    result = []
    keys = 0
    num = 0
    # 获取随机点
    # points = get_points()
    tags = tag
    #
    # print(points)
    inf = 1e9
    #
    plane = sg.Polygon([(inf, inf), (inf, -inf), (-inf, -inf), (-inf, inf)])
    # print(plane)

    plots = [  # ('scatter', ScatterVisitor(), 0, False, '', 10),
        # ('tree-1', TreeVisitor(), 1, True, '', 10),
        # ('tree-2', TreeVisitor(), 2, True, '', 10),
        # ('tree-3', TreeVisitor(), 3, True, '', 10),
        ('tree-full', TreeVisitor(), 9999, True, '', leaf_sizes)]

    for tag, visitor, max_splits, draw_splits, seed, leaf_size in plots:
        fn = tag + str(num_pic) + '.png'
        print(fn, '...')
        # fig： matplotlib.figure.Figure 对象
        # ax：子图对象（ matplotlib.axes.Axes）或者是他的数组
        # plt.subplots() 返回一个 Figure实例fig 和一个 AxesSubplot实例ax 。这个很好理解，fig代表整个图像，ax代表坐标轴和画的图
        fig, ax = plt.subplots()  # 画布

        fig.set_size_inches(16, 9)  # 设置大小
        # 建立图graph
        # import pygraphviz as pyg
        # g=pyg.AGraph()  #建立图
        # g.add_node('A')  #建立点
        # g.add_edge('A','B')  #建立边
        # g.add_edge('A','C')  #建立边
        # g.layout(prog='dot')  #绘图类型
        # g.draw('pyg1.png')   #绘制

        graph = pygraphviz.AGraph()
        dim = 2  # 维数
        layer = 1  # 生成树的层
        split_points(extract_data_num, layer, dim, ax, graph, plane, points, range(len(points)), visitor=visitor,
                     max_splits=max_splits, draw_splits=draw_splits, seed=seed, leaf_size=leaf_size)

        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        fig.savefig('pic/'+fn, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)

        graph.layout(prog='dot')
        graph.draw('pic/'+tag + str(num_pic) + '-graphviz.png')
        plt.show()
        return result


# 把数据按组进行排序，并和标签分离，方便后面处理
def partData(space_point, dim=2):
    space_point.sort(key=lambda x: x[dim + 1], reverse=False)  # 按升序排序
    ar = np.array(space_point)
    lens = len(space_point)
    data = []  # 数据点
    for i in range(lens):
        data.append(i)
        data[i] = []
        for j in range(dim):
            data[i].append(space_point[i][j])
    label = ar[:, dim]  # 类别标签
    leaves = ar[:, dim + 1]  # 叶子节点标签
    layer = ar[:, dim + 2]  # 所在层数

    return data, label.tolist(), leaves.tolist(), layer.tolist()


if __name__ == '__main__':
    filepath = 'D:\\PycharmObject\\dataset\\test\\2d-3c.arff'
    data, flag = loadData(filepath)
    k = 10
    extract_data_num = dataExtract(data, k)
    space_point = space(data, flag, extract_data_num)
    print(space_point)
    data, label, leaves, layer = partData(space_point, 2)
    # print(data)
    # print(leaves)
    # print(layer)
    # print(len(layer))

    # 测试ANNOY后标签是否出错
    # filepath = 'D:\\PycharmObject\\dataset\\test\\2d-3c.arff'
    # data, flag = loadData(filepath)
    # #print(flag)
    # unique1, counts1 = np.unique(flag, return_counts=True)
    # print(unique1)
    # print(counts1)
    # filepath = "D:\\PycharmObject\\dataset\\test\\2d-3c.arff"
    # space_point = space(filepath)
    # #print(space_point)
    # ar = np.array(space_point)
    # #print(ar[:, 2])
    # unique2, counts2 = np.unique(ar[:, 2], return_counts=True)#提取出标签列的数据
    # print(unique2)
    # print(counts2)

    # 检验层数是否正确
    # nums = 0
    # for i in range(len(space_point)):
    #     if space_point[i][4] == 6:
    #         nums = nums + 1
    # print(nums) #


