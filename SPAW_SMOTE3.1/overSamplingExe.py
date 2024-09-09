from densityAndCenter import dataExtract, data_den_order
from dataReading.datasetKeel import loadData
from dataset.dataProcessing import data_read_keel, data_read_arff, data_read_keel_np,data_ablone_str_to_num
from treeSpacePartingDim import space, data_str_to_flow
from treeSpacePartingDim import partData, data_str_to_flow
from minClassLabel import select_label_statistics, select_two_label_statistics
from generateData import generateData, generateDataTest
import numpy as np
from dataReading.datasetXls import writeData, loadDataMid, writeDataTest, writeDataReTest
import pandas as pd
from treeSmote import region_smote, test
from classifierEvaluation import evaluation_kfold,evaluation_division, evaluation_kfold_f1, evaluation_kfold_recall, evaluation_kfold_roc_auc, write_evaluation, evaluation_kfold_Gmean, evaluation_kfold_auc_f1_gmean
from downSampling import remove_maj_point, TomekLinks_test
from removeNoise import re_noise, searchKNN_index
import matplotlib.pyplot as plt
from dataPaint import show_dataset, show_clustered_dataset
from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE, RandomOverSampler, BorderlineSMOTE
import smote_variants as sv
import os
from collections import Counter



#filepath = "D:\\PycharmObject\\dataset\\test\\2d-3c.arff"
# filepath = 'D:\\PycharmObject\\annoy\\dataset\\Syn\\syn2.arff'
# data, tag = loadData(filepath)
def maj_min_ratio(label):
    min_num = 0
    for i in range(len(label)):
        if label[i] == 1:
            min_num = min_num + 1

    return (len(label)-min_num)/min_num, min_num

def dataAnalyze(label):
    count = Counter(label)
    print(count)





if __name__ == '__main__':
    #是否调用中间结果
    for root, dirs, files in os.walk(r"E:\\PyCharmProject\\SPWSYN3.1\\dataset\\KEEL\\test"):
        k = 3
        ratio = 0.6
        leaf_sizes = 10
        # count_result_Min = []
        # count_result_Max = []
        # kmean_flag = 1
        result_auc = []
        result_f1 = []
        # result_recall = []
        result_Gmean = []
        result_all = []
        result_key = 0
        result_key_all = 0
        for file in files:
            result_auc.append(result_key)
            result_auc[result_key] = []
            result_auc[result_key].append(str(file))
            result_f1.append(result_key)
            result_f1[result_key] = []
            result_f1[result_key].append(str(file))
            result_Gmean.append(result_key)
            result_Gmean[result_key] = []
            result_Gmean[result_key].append(str(file))
            result_key_all_x = result_key_all
            for i in range(3):
                result_all.append(result_key_all_x)
                result_all[result_key_all_x] = []
                result_all[result_key_all_x].append(str(file))
                result_key_all_x = result_key_all_x + 1
            # result_recall.append(result_key)
            # result_recall[result_key] = []
            # result_recall[result_key].append(str(file))
            #data, tag = generateDataTest(sam_Min, sam_Max)
            #ecoli2
            #glass1
            #cleveland

            filepath = os.path.join(root, file)
            print(filepath)
            data_ori, tag_ori, data_ori_np, tag_ori_np = data_read_keel_np(filepath)
            # data_ori, tag_ori, data_ori_np, tag_ori_np = data_ablone_str_to_num(filepath)
            #print(data_ori)
            #print(data_ori_np)

            # filepath = 'D:\\PycharmObject\\ZQannoy3.3\\dataset\\Syn\\jain.arff'
            # data_ori, tag_ori = data_read_arff(filepath)
            #print(tag_ori)
            data = data_str_to_flow(data_ori)
            tag = tag_ori
            #print(type(data))
            # data_ori = []
            # tag_ori = []
            # data_ori_np = np.array(data).astype('float64')
            # tag_ori_np = np.array(tag).astype('float64')
            dim = len(data_ori[0])

            # （算法核心，需要划分的高密度点）extract_data_num 高密度点所在下标
            # re_k = 5
            # pointKNN_remove = searchKNN(data_ori_np, re_k)
            ir_ratio_ori, data_min_num_ori = maj_min_ratio(tag)
            result_auc[result_key].append(ir_ratio_ori)
            result_f1[result_key].append(ir_ratio_ori)
            result_Gmean[result_key].append(ir_ratio_ori)

            #ir_ratio = ir_ratio_ori
            # data, tag, data_min_num, ir_ratio = re_noise(data_ori_np, tag_ori, pointKNN_remove, re_k)
            extract_data_num,data_spreadability = dataExtract(data_ori_np, k, ratio)
            #ir_ratio_int = int(ir_ratio)
            #print(pointKNN_remove)
            # print(extract_data_num)
            # 空间划分

            space_point, proportion = space(leaf_sizes, data_ori_np, tag, dim, extract_data_num, 0, data_spreadability)
            proportion_sum = sum(proportion)
            # print(proportion_sum)
            proportion_standard = []
            need_add_min = len(tag) - data_min_num_ori - data_min_num_ori
            need_add_min = need_add_min * 1
            for pro_num in range(len(proportion)):
                proportion_standard.append(proportion[pro_num]/proportion_sum*need_add_min)
            print(proportion_standard)
            data_mid, label_mid = test(space_point, dim, proportion_standard, data_ori_np, tag_ori)



            # #去噪
            # re_k = 3
            # re_w = 1
            # pointKNN_remove = searchKNN_index(data_mid, re_k)
            # data_re, label_re, data_min_num_re = re_noise(data_mid, label_mid, pointKNN_remove, re_k, re_w)
            #
            # #去多数类
            # need_remove_maj_num = len(label_re) - data_min_num_re - data_min_num_re
            # density_order = data_den_order(data_re)
            # datas, tags = remove_maj_point(data_re, label_re, density_order, need_remove_maj_num)

            #不去噪，不去多数类
            new_data = data_mid
            new_label = label_mid

            # new_data = datas
            # new_label = tags

            # new_min_num = 0
            # for np in range(len(new_label)):
            #     if new_label[np] == 1:
            #         new_min_num = new_min_num + 1
            # print("最终有样本 {}".format(len(new_label)))
            # print("New_IR is {}".format((len(new_label) - new_min_num) / new_min_num))
            dataAnalyze(new_label)


            # ROS = sv.Random_SMOTE()
            # x_ROS_resampled, y_ROS_resampled = ROS.sample(data_ori_np, tag_ori_np)
            smote_only = sv.SMOTE()
            x_smote_resampled, y_smote_resampled = smote_only.fit_resample(data_ori_np, tag_ori_np)
            BL1 = sv.Borderline_SMOTE1()
            x_BL1_resampled, y_BL1_resampled = BL1.fit_resample(data_ori_np, tag_ori_np)
            BL2 = sv.Borderline_SMOTE2()
            x_BL2_resampled, y_BL2_resampled = BL2.fit_resample(data_ori_np, tag_ori_np)
            ADA = sv.ADASYN()
            x_ADA_resampled, y_ADA_resampled = ADA.fit_resample(data_ori_np, tag_ori_np)
            # KM = sv.kmeans_SMOTE()
            # x_KM_resampled, y_KM_resampled = KM.fit_resample(data_ori_np, tag_ori_np)
            # SMOTETomek = sv.SMOTE_TomekLinks()
            # x_STom_resampled, y_STom_resampled = SMOTETomek.fit_resample(data_ori_np, tag_ori_np)
            KADASYN = sv.KernelADASYN()
            x_KADA_resampled, y_KADA_resampled = KADASYN.fit_resample(data_ori_np, tag_ori_np)
            PWS = sv.ProWSyn()
            x_PWS_resampled, y_PWS_resampled = PWS.fit_resample(data_ori_np, tag_ori_np)
            PFS = sv.polynom_fit_SMOTE()
            x_PFS_resampled, y_PFS_resampled = PFS.fit_resample(data_ori_np, tag_ori_np)
            # AMSCO = sv.AMSCO()
            # x_AMSCO_resampled, y_AMSCO_resampled = AMSCO.fit_resample(data_ori_np, tag_ori_np)
            # ANS = sv.ANS()
            # x_ANS_resampled, y_ANS_resampled = ANS.fit_resample(data_ori_np, tag_ori_np)
            CCR = sv.CCR()
            x_CCR_resampled, y_CCR_resampled = CCR.fit_resample(data_ori_np, tag_ori_np)
            # SMOTEENN = sv.SMOTE_ENN()
            # x_SMOTEENN_resampled, y_SMOTEENN_resampled = SMOTEENN.fit_resample(data_ori_np, tag_ori_np)



            #记录结果

            # # print('ROS: ', end='')
            # result_auc[result_key].append(evaluation_kfold_roc_auc(x_ROS_resampled, y_ROS_resampled))
            # result_f1[result_key].append(evaluation_kfold_f1(x_ROS_resampled, y_ROS_resampled))
            # result_Gmean[result_key].append(evaluation_kfold_Gmean(x_ROS_resampled, y_ROS_resampled))
            # # result_recall[result_key].append(evaluation_kfold_recall(x_ROS_resampled, y_ROS_resampled))
            #
            # # print('smote: ', end='')
            # result_auc[result_key].append(evaluation_kfold_roc_auc(x_smote_resampled, y_smote_resampled))
            # result_f1[result_key].append(evaluation_kfold_f1(x_smote_resampled, y_smote_resampled))
            # result_Gmean[result_key].append(evaluation_kfold_Gmean(x_smote_resampled, y_smote_resampled))
            # # result_recall[result_key].append(evaluation_kfold_recall(x_smote_resampled, y_smote_resampled))
            #
            # # print('BL1: ', end='')
            # result_auc[result_key].append(evaluation_kfold_roc_auc(x_BL_resampled, y_BL_resampled))
            # result_f1[result_key].append(evaluation_kfold_f1(x_BL_resampled, y_BL_resampled))
            # result_Gmean[result_key].append(evaluation_kfold_Gmean(x_BL_resampled, y_BL_resampled))
            # # result_recall[result_key].append(evaluation_kfold_recall(x_BL_resampled, y_BL_resampled))
            #
            # # print('ADA: ', end='')
            # result_auc[result_key].append(evaluation_kfold_roc_auc(x_ADA_resampled, y_ADA_resampled))
            # result_f1[result_key].append(evaluation_kfold_f1(x_ADA_resampled, y_ADA_resampled))
            # result_Gmean[result_key].append(evaluation_kfold_Gmean(x_ADA_resampled, y_ADA_resampled))
            # # result_recall[result_key].append(evaluation_kfold_recall(x_ADA_resampled, y_ADA_resampled))
            #
            # # print('KM: ', end='')
            # result_auc[result_key].append(evaluation_kfold_roc_auc(x_KM_resampled, y_KM_resampled))
            # result_f1[result_key].append(evaluation_kfold_f1(x_KM_resampled, y_KM_resampled))
            # result_Gmean[result_key].append(evaluation_kfold_Gmean(x_KM_resampled, y_KM_resampled))
            # # result_recall[result_key].append(evaluation_kfold_recall(x_KM_resampled, y_KM_resampled))
            #
            # # print('PWS: ', end='')
            # result_auc[result_key].append(evaluation_kfold_roc_auc(x_PWS_resampled, y_PWS_resampled))
            # result_f1[result_key].append(evaluation_kfold_f1(x_PWS_resampled, y_PWS_resampled))
            # result_Gmean[result_key].append(evaluation_kfold_Gmean(x_PWS_resampled, y_PWS_resampled))
            # # result_recall[result_key].append(evaluation_kfold_recall(x_PWS_resampled, y_PWS_resampled))
            #
            # # print('PFS: ', end='')
            # result_auc[result_key].append(evaluation_kfold_roc_auc(x_PFS_resampled, y_PFS_resampled))
            # result_f1[result_key].append(evaluation_kfold_f1(x_PFS_resampled, y_PFS_resampled))
            # result_Gmean[result_key].append(evaluation_kfold_Gmean(x_PFS_resampled, y_PFS_resampled))
            # # result_recall[result_key].append(evaluation_kfold_recall(x_PFS_resampled, y_PFS_resampled))
            #
            # # print('SMOTETomek: ', end='')
            # result_auc[result_key].append(evaluation_kfold_roc_auc(x_STom_resampled, y_STom_resampled))
            # result_f1[result_key].append(evaluation_kfold_f1(x_STom_resampled, y_STom_resampled))
            # result_Gmean[result_key].append(evaluation_kfold_Gmean(x_STom_resampled, y_STom_resampled))
            # # result_recall[result_key].append(evaluation_kfold_recall(x_STom_resampled, y_STom_resampled))
            #
            # # print('SMOTEENN: ', end='')
            # result_auc[result_key].append(evaluation_kfold_roc_auc(x_SMOTEENN_resampled, y_SMOTEENN_resampled))
            # result_f1[result_key].append(evaluation_kfold_f1(x_SMOTEENN_resampled, y_SMOTEENN_resampled))
            # result_Gmean[result_key].append(evaluation_kfold_Gmean(x_SMOTEENN_resampled, y_SMOTEENN_resampled))
            # # result_recall[result_key].append(evaluation_kfold_recall(x_SMOTEENN_resampled, y_SMOTEENN_resampled))
            #
            # # print("zqsmote: ", end='')
            # result_auc[result_key].append(evaluation_kfold_roc_auc(new_data, new_label))
            # result_f1[result_key].append(evaluation_kfold_f1(new_data, new_label))
            # result_Gmean[result_key].append(evaluation_kfold_Gmean(new_data, new_label))
            # # result_recall[result_key].append(evaluation_kfold_recall(new_data, new_label))
            # #print(new_data)
            # # print(new_label)

            #第二个方法记录结果

            # r_auc, r_f1, r_gmean = evaluation_kfold_auc_f1_gmean(x_ROS_resampled, y_ROS_resampled)
            # result_auc[result_key].append(r_auc)
            # result_f1[result_key].append(r_f1)
            # result_Gmean[result_key].append(r_gmean)

            r_auc, r_f1, r_gmean = evaluation_kfold_auc_f1_gmean(x_smote_resampled, y_smote_resampled)
            result_auc[result_key].append(r_auc)
            result_f1[result_key].append(r_f1)
            result_Gmean[result_key].append(r_gmean)
            result_all[result_key_all].append(r_gmean)
            result_all[result_key_all+1].append(r_f1)
            result_all[result_key_all+2].append(r_auc)

            r_auc, r_f1, r_gmean = evaluation_kfold_auc_f1_gmean(x_BL1_resampled, y_BL1_resampled)
            result_auc[result_key].append(r_auc)
            result_f1[result_key].append(r_f1)
            result_Gmean[result_key].append(r_gmean)
            result_all[result_key_all].append(r_gmean)
            result_all[result_key_all + 1].append(r_f1)
            result_all[result_key_all + 2].append(r_auc)

            r_auc, r_f1, r_gmean = evaluation_kfold_auc_f1_gmean(x_BL2_resampled, y_BL2_resampled)
            result_auc[result_key].append(r_auc)
            result_f1[result_key].append(r_f1)
            result_Gmean[result_key].append(r_gmean)
            result_all[result_key_all].append(r_gmean)
            result_all[result_key_all + 1].append(r_f1)
            result_all[result_key_all + 2].append(r_auc)

            r_auc, r_f1, r_gmean = evaluation_kfold_auc_f1_gmean(x_ADA_resampled, y_ADA_resampled)
            result_auc[result_key].append(r_auc)
            result_f1[result_key].append(r_f1)
            result_Gmean[result_key].append(r_gmean)
            result_all[result_key_all].append(r_gmean)
            result_all[result_key_all + 1].append(r_f1)
            result_all[result_key_all + 2].append(r_auc)



            # r_auc, r_f1, r_gmean = evaluation_kfold_auc_f1_gmean(x_STom_resampled, y_STom_resampled)
            # result_auc[result_key].append(r_auc)
            # result_f1[result_key].append(r_f1)
            # result_Gmean[result_key].append(r_gmean)

            r_auc, r_f1, r_gmean = evaluation_kfold_auc_f1_gmean(x_KADA_resampled, y_KADA_resampled)
            result_auc[result_key].append(r_auc)
            result_f1[result_key].append(r_f1)
            result_Gmean[result_key].append(r_gmean)
            result_all[result_key_all].append(r_gmean)
            result_all[result_key_all + 1].append(r_f1)
            result_all[result_key_all + 2].append(r_auc)

            r_auc, r_f1, r_gmean = evaluation_kfold_auc_f1_gmean(x_PWS_resampled, y_PWS_resampled)
            result_auc[result_key].append(r_auc)
            result_f1[result_key].append(r_f1)
            result_Gmean[result_key].append(r_gmean)
            result_all[result_key_all].append(r_gmean)
            result_all[result_key_all + 1].append(r_f1)
            result_all[result_key_all + 2].append(r_auc)

            r_auc, r_f1, r_gmean = evaluation_kfold_auc_f1_gmean(x_PFS_resampled, y_PFS_resampled)
            result_auc[result_key].append(r_auc)
            result_f1[result_key].append(r_f1)
            result_Gmean[result_key].append(r_gmean)
            result_all[result_key_all].append(r_gmean)
            result_all[result_key_all + 1].append(r_f1)
            result_all[result_key_all + 2].append(r_auc)

            # r_auc, r_f1, r_gmean = evaluation_kfold_auc_f1_gmean(x_AMSCO_resampled, y_AMSCO_resampled)
            # result_auc[result_key].append(r_auc)
            # result_f1[result_key].append(r_f1)
            # result_Gmean[result_key].append(r_gmean)
            # result_all[result_key_all].append(r_gmean)
            # result_all[result_key_all + 1].append(r_f1)
            # result_all[result_key_all + 2].append(r_auc)

            # r_auc, r_f1, r_gmean = evaluation_kfold_auc_f1_gmean(x_ANS_resampled, y_ANS_resampled)
            # result_auc[result_key].append(r_auc)
            # result_f1[result_key].append(r_f1)
            # result_Gmean[result_key].append(r_gmean)
            # result_all[result_key_all].append(r_gmean)
            # result_all[result_key_all + 1].append(r_f1)
            # result_all[result_key_all + 2].append(r_auc)

            r_auc, r_f1, r_gmean = evaluation_kfold_auc_f1_gmean(x_CCR_resampled, y_CCR_resampled)
            result_auc[result_key].append(r_auc)
            result_f1[result_key].append(r_f1)
            result_Gmean[result_key].append(r_gmean)
            result_all[result_key_all].append(r_gmean)
            result_all[result_key_all + 1].append(r_f1)
            result_all[result_key_all + 2].append(r_auc)

            # r_auc, r_f1, r_gmean = evaluation_kfold_auc_f1_gmean(x_KM_resampled, y_KM_resampled)
            # result_auc[result_key].append(r_auc)
            # result_f1[result_key].append(r_f1)
            # result_Gmean[result_key].append(r_gmean)
            # result_all[result_key_all].append(r_gmean)
            # result_all[result_key_all + 1].append(r_f1)
            # result_all[result_key_all + 2].append(r_auc)



            # r_auc, r_f1, r_gmean = evaluation_kfold_auc_f1_gmean(x_SMOTEENN_resampled, y_SMOTEENN_resampled)
            # result_auc[result_key].append(r_auc)
            # result_f1[result_key].append(r_f1)
            # result_Gmean[result_key].append(r_gmean)

            r_auc, r_f1, r_gmean = evaluation_kfold_auc_f1_gmean(new_data, new_label)
            result_auc[result_key].append(r_auc)
            result_f1[result_key].append(r_f1)
            result_Gmean[result_key].append(r_gmean)
            result_all[result_key_all].append(r_gmean)
            result_all[result_key_all + 1].append(r_f1)
            result_all[result_key_all + 2].append(r_auc)





            result_key = result_key + 1
            result_key_all = result_key_all + 3


        print(result_Gmean)
        write_evaluation(result_auc, "E:\\PyCharmProject\\SPWSYN3.1\\evaluation\\keel_test_auc.xlsx")
        write_evaluation(result_f1, "E:\\PyCharmProject\\SPWSYN3.1\\evaluation\\keel_test_f1.xlsx")
        write_evaluation(result_Gmean, "E:\\PyCharmProject\\SPWSYN3.1\\evaluation\\keel_test_Gmean.xlsx")
        write_evaluation(result_all,"E:\\PyCharmProject\\SPWSYN3.1\\evaluation\\keel_test_all.xlsx")
        #write_evaluation(result_recall, "D:\\PycharmObject\\ZQannoy5.0\\evaluation\\keel_h_recall.xlsx")











