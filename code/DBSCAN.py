import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from time import time
from sklearn.cluster import DBSCAN
import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import time

def get_data(file):

    GPSLon = []
    GPSLat = []
    tag = []
    # t1 = time()
    for picindex in range(len(file)):
        data1 = pd.read_excel(all_path + file[picindex], header=0)
        data = data1.loc[:, ['lon', 'lat', 'tag']]
        GPSLat.append(data['lat'])
        GPSLon.append(data['lon'])
        tag.append(data['tag'])

    feature = []
    taglist = []
    for picindex in range(len(file)):

        x = GPSLon[picindex]
        y = GPSLat[picindex]
        onetag = tag[picindex]
        # dbscan聚类方式
        dataSet = [[0 for j in range(2)] for h in range(len(x))]
        for j in range(len(x)):
            dataSet[j][0] = x[j]
            dataSet[j][1] = y[j]
        feature.append(dataSet)
        taglist.append(onetag)
    return feature, taglist

if __name__ == '__main__':
    # 加载数据
    acc_list=[]
    test_road_precision = 0
    test_road_recall = 0
    test_road_f1score = 0
    test_field_precision = 0
    test_field_recall = 0
    test_field_f1score = 0
    test_accuracy = 0
    test_macro_precision = 0
    test_macro_recall = 0
    test_macro_f1score = 0
    test_weight_precision = 0
    test_weight_recall = 0
    test_weight_f1score = 0
    kfold = 10
    lenset = kfold
    time_start = time.time()
    all_file = []  # 文件夹下的所有文件
    print('loading data...')
    path='' #分成10折所在的文件夹
    all_path="" #总文件所在文件夹
    print('USE data:',path)
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    for filepath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            all_file.append(filename)
    target_names = ['road', 'field']
    for kfold in range(kfold):
        #训练测试文件划分
        # train_file, test_file, _, _ = train_test_split(all_file, all_file, test_size=0.1, random_state= kfold)
        train_file=os.listdir(path + str(kfold) + "/train")
        train_file.extend(os.listdir(path + str(kfold) + "/val"))
        test_file=os.listdir(path + str(kfold) + "/test")
        # DBSCAN聚类
        # 多进程统计数组
        train_data, train_tag = get_data(train_file)
        print('X_train:', len(train_data))
        final_result = -1
        map_eps = -1
        map_minpts = -1
        for eps in range(1, 52, 2):#30
            for minpts in range(1, 210, 10):#1010

                dbscan = DBSCAN(eps=eps / 100000, min_samples=minpts)
                combin_tag = []
                pred_list = []
                for trainindex in range(len(train_data)):
                    clusteringresult = dbscan.fit(train_data[trainindex])
                    nparraylabel = np.array(clusteringresult.labels_)
                    label0_1 = np.where(nparraylabel >= 0, 1, 0)
                    clustering = label0_1.tolist()
                    combin_tag += train_tag[trainindex].tolist()
                    pred_list += clustering

                macro_f1score = f1_score(combin_tag, pred_list, average='macro')
                if macro_f1score > final_result:
                    final_result = macro_f1score
                    map_eps = eps
                    map_minpts = minpts
        print('kfold: ',kfold)
        print('macro_f1score: ', final_result)
        print('eps: ', map_eps)
        print('minpts: ', map_minpts)

        # total_macro_f1score += final_result

        X_test, y_test = get_data(test_file) #获取测试数据集
        print('X_test:',len(X_test))
        dbscan = DBSCAN(eps=map_eps / 100000, min_samples=map_minpts)
        test_tag = []
        test_pred = []
        for testindex in range(len(X_test)):
            clusteringresult = dbscan.fit(X_test[testindex])
            nparraylabel = np.array(clusteringresult.labels_)
            label0_1 = np.where(nparraylabel >= 0, 1, 0)
            clustering = label0_1.tolist()
            test_pred += clustering
            test_tag += y_test[testindex].tolist()

        # result = classification_report(test_tag, test_pred, digits=10, target_names=target_names,
        #                                output_dict=True)
        acc_list.append(classification_report(test_tag, test_pred, digits=4, output_dict=True)['accuracy'])

        test_road_precision += classification_report(test_tag, test_pred, digits=4, output_dict=True)['0'][
            'precision']
        test_road_recall += classification_report(test_tag, test_pred, digits=4, output_dict=True)['0']['recall']
        test_road_f1score += classification_report(test_tag, test_pred, digits=4, output_dict=True)['0']['f1-score']
        test_field_precision += classification_report(test_tag, test_pred, digits=4, output_dict=True)['1'][
            'precision']
        test_field_recall += classification_report(test_tag, test_pred, digits=4, output_dict=True)['1']['recall']
        test_field_f1score += classification_report(test_tag, test_pred, digits=4, output_dict=True)['1']['f1-score']
        test_accuracy += classification_report(test_tag, test_pred, digits=4, output_dict=True)['accuracy']
        test_macro_precision += classification_report(test_tag, test_pred, digits=4, output_dict=True)['macro avg'][
            'precision']
        test_macro_recall += classification_report(test_tag, test_pred, digits=4, output_dict=True)['macro avg'][
            'recall']
        test_macro_f1score += classification_report(test_tag, test_pred, digits=4, output_dict=True)['macro avg'][
            'f1-score']
        test_weight_precision += \
            classification_report(test_tag, test_pred, digits=4, output_dict=True)['weighted avg'][
                'precision']
        test_weight_recall += classification_report(test_tag, test_pred, digits=4, output_dict=True)['weighted avg'][
            'recall']
        test_weight_f1score += classification_report(test_tag, test_pred, digits=4, output_dict=True)['weighted avg'][
            'f1-score']

    print(acc_list)
    print('               precision    recall    f1score')
    print('        0    ' + str(round(test_road_precision / lenset, 4)) + '      ' + str(
        round(test_road_recall / lenset, 4)) + '    ' + str(round(test_road_f1score / lenset, 4)))
    print('        1    ' + str(round(test_field_precision / lenset, 4)) + '      ' + str(
        round(test_field_recall / lenset, 4)) + '    ' + str(round(test_field_f1score / lenset, 4)))
    print('\n')
    print('    accuracy                          ' + str(round(test_accuracy / lenset, 4)))
    print('   macro avg    ' + str(round(test_macro_precision / lenset, 4)) + '      ' + str(
        round(test_macro_recall / lenset, 4)) + '    ' + str(round(test_macro_f1score / lenset, 4)))
    print('weighted avg    ' + str(round(test_weight_precision / lenset, 4)) + '      ' + str(
        round(test_weight_recall / lenset, 4)) + '    ' + str(round(test_weight_f1score / lenset, 4)))
