import numpy as np
import math
#note ground truth is vector, rank_list is the sorted item index.

def label2vec(label_list, input_size):
    #label_list -> list
    #input_size -> item number
    label_vec = np.zeros(input_size)
    for label in label_list:
        label_vec[label]=1
    return label_vec

def get_repeat_explore(repeat_list, pred_rank_list, k):
    count = 0
    repeat_cnt = 0.0
    for pred in pred_rank_list:
        if count >= k:
            break
        if pred in repeat_list:
            repeat_cnt += 1
        count += 1
    repeat_ratio = repeat_cnt/k
    return repeat_ratio, 1-repeat_ratio

def get_DCG(truth_list, pred_rank_list, k):
    count = 0
    dcg = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if pred in truth_list:
            dcg += (1)/math.log2(count+1+1)
        count += 1
    return dcg

def get_NDCG(truth_list, pred_rank_list, k):
    dcg = get_DCG(truth_list, pred_rank_list, k)
    idcg = 0
    num_item = len(truth_list)
    for i in range(num_item):
        idcg += (1) / math.log2(i + 1 + 1)
    ndcg = dcg / idcg
    return ndcg

def get_HT(truth_list, pred_rank_list, k):
    count = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if pred in truth_list:
            return 1
        count += 1
    return 0

def get_Recall(truth_list, pred_rank_list, k):
    truth_num = len(truth_list)
    count = 0
    correct = 0.0
    for pred in pred_rank_list:
        if count >= k:
            break
        if pred in truth_list:
            correct += 1
        count += 1
    recall = correct/truth_num
    return recall


def get_precision_recall_Fscore(groundtruth, pred):
    a = groundtruth
    b = pred
    correct = 0
    truth = 0
    positive = 0

    for idx in range(len(a)):
        if a[idx] == 1:
            truth += 1
            if b[idx] == 1:
                correct += 1
        if b[idx] == 1:
            positive += 1

    flag = 0
    if 0 == positive:
        precision = 0
        flag = 1
        #print('postivie is 0')
    else:
        precision = correct/positive
    if 0 == truth:
        recall = 0
        flag = 1
        #print('recall is 0')
    else:
        recall = correct/truth

    if flag == 0 and precision + recall > 0:
        F = 2*precision*recall/(precision+recall)
    else:
        F = 0
    return precision, recall, F, correct