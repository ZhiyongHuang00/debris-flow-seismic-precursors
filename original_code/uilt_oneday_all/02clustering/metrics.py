import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score ,v_measure_score, adjusted_mutual_info_score, accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

nmi = normalized_mutual_info_score
ari = adjusted_rand_score
accuracy = accuracy_score

def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

# def tracker(label,pred,n_clusters):
#     """"用于跟踪聚类结果，并返回真实标签、k-means 预测标签和处理后的预测标签。"""
#     ###
#     label_true = []
#     label_pred_km = []
#     label_pred_last = []
#
#     if len(label) != len(pred):
#         print("size error {0} != {1}".format(len(label),len(pred)))
#     for i,classnum in enumerate(label):
#         if classnum == 0:
#             label_true.append(0)
#         elif classnum == 1:
#             label_true.append(1)
#         elif classnum == 2:
#             label_true.append(2)
#         elif classnum == 3:
#             label_true.append(3)
#     for i,classnum in enumerate(pred):
#         if label[i] != 'Check':
#             label_pred_km.append(classnum)
#
#
#     label_pred = np.copy(label_pred_km)
#
#     ######
#     for i in range(n_clusters):
#         check = []
#         for n,classnum in enumerate(pred):
#             if classnum == i and label[n] != 'Check':  #提取第i个簇的,并得到其中的点对应的真实标签
#                 check.append(label[n])
#         if len(check) > 0 :
#             """0DF、1EQ、2NS、3RF"""
#             print ("in cluster{0} : DFnum={1},EQnum={2},NSnum={3},RFnum={4}".format(i,check.count(0),check.count(1),check.count(2),check.count(3)))
#             if max(check,key=check.count) == 0:
#                 print ("clster{}->DFclass".format(i))
#                 label_pred = [0 if x==i else x for x in label_pred]
#             elif max(check,key=check.count) == 1:
#                 print ("clster{}->EQclass".format(i))
#                 label_pred = [1 if x==i else x for x in label_pred]
#             elif max(check,key=check.count) == 2:
#                 print ("clster{}->NSclass".format(i))
#                 label_pred = [2 if x==i else x for x in label_pred]
#             elif max(check,key=check.count) == 3:
#                 print ("clster{}->RFclass".format(i))
#                 label_pred = [3 if x==i else x for x in label_pred]
#         else :print ("no tracker in cluster="+str(i))
#
#     return np.array(label_true),np.array(label_pred),np.array(label_pred_km)
def tracker(label, pred, n_clusters):
    """
    用于跟踪聚类结果，并返回真实标签、k-means 预测标签和处理后的预测标签。
    """
    label_true = []
    label_pred_km = []
    label_pred_last = []

    if len(label) != len(pred):
        print("size error {0} != {1}".format(len(label), len(pred)))
        return None, None, None

    # 处理真实标签
    for classnum in label:
        if classnum == 0:
            label_true.append(0)
        elif classnum == 1:
            label_true.append(1)
        elif classnum == 2:
            label_true.append(2)
        elif classnum == 3:
            label_true.append(3)

    # 处理k-means预测标签
    for i, classnum in enumerate(pred):
        if label[i] != 'Check':
            label_pred_km.append(classnum)

    # label_pred = np.copy(label_pred_km)
    label_pred = np.copy(pred)

    # 处理聚类结果并进行标签映射
    for i in range(n_clusters):
        check = []
        for n, classnum in enumerate(pred):
            if classnum == i and label[n] != 'Check':  # 提取第i个簇的点，并得到其中的点对应的真实标签
                check.append(label[n])
        """1DF、2EQ、0NS、3RF"""

        if len(check) > 0:
            print("in cluster{0} : DFnum={1}, EQnum={2}, NSnum={3}, RFnum={4}".format(i, check.count(0), check.count(1), check.count(2), check.count(3)))
            most_common_label = max(set(check), key=check.count)
            if most_common_label == 1:
                print("cluster{}->DFclass".format(i))
                label_pred = ["DF" if x == i else x for x in label_pred]
            elif most_common_label == 2:
                print("cluster{}->EQclass".format(i))
                label_pred = ["EQ" if x == i else x for x in label_pred]
            elif most_common_label == 0:
                print("cluster{}->NSclass".format(i))
                label_pred = ["NS" if x == i else x for x in label_pred]
            elif most_common_label == 3:
                print("cluster{}->RFclass".format(i))
                label_pred = ["RF" if x == i else x for x in label_pred]
        else:
            print("no tracker in cluster=" + str(i))
    label_pred_end = []
    """1DF、2EQ、0NS、3RF"""
    for classnum in label_pred:
        if classnum == "DF":
            label_pred_end.append(1)
        elif classnum == "EQ":
            label_pred_end.append(2)
        elif classnum == "NS":
            label_pred_end.append(0)
        elif classnum == "RF":
            label_pred_end.append(3)
    return np.array(label_true), np.array(label_pred_end), np.array(label_pred_km)

def Clustering_metrics(label,pred,n_clusters,datasettype='check') :
    """用于计算聚类模型的性能指标，
    包括归一化互信息（NMI）、调整兰德指数（ARI）和纯度（Purity）"""

    lists = ["nmi","ari","purity"]
    truelist,predlist,predlist_km = tracker(label,pred,n_clusters,datasettype='check')
    nmi = nmi(truelist,predlist_km)
    ari = ari(truelist,predlist_km)
    purity = purity_score(truelist,predlist_km)
    metric=[nmi,ari,purity]
    lists.append(metric)

    return lists
