import smote_variants as sv
from imblearn.under_sampling import TomekLinks


def remove_maj_point(point, label, maj_order, need_remove_maj_num):
    filter_point = []
    filter_label = []
    if need_remove_maj_num == 0:
        #print("降采样了0个样本")
        return point, label

    else:
        #print(need_remove_maj_num)
        k = need_remove_maj_num
        # k_maj = len(label)
        # if k > k_maj*0.3:
        #     k = int(k_maj*0.3)
        #print(len(maj_ind))
        #print(len(label))
        for i in range(len(label)):
            j = maj_order[i]
            if label[j] == 0 and k > 0:
                k = k-1
                label[j] = -1

        for i in range(len(label)):
            if label[i] != -1:
                filter_point.append(point[i])
                filter_label.append(label[i])



        print("降采样了{0}个样本".format(need_remove_maj_num-k))

        return filter_point, filter_label

def TomekLinks_test(point, label):
    TL = TomekLinks()
    filter_point, filter_label = TL.fit_resample(point, label)
    print("降采样了{0}个样本".format(len(point) - len(filter_label)))
    return filter_point, filter_label




