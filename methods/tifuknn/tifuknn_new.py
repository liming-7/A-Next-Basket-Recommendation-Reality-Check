from __future__ import unicode_literals, print_function, division
import numpy as np
import sys
import math
import csv

import os
import json

from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

topk = 10

def group_history_list(his_list, group_size):
    grouped_vec_list = []
    if len(his_list) < group_size:
        for j in range(len(his_list)):
            grouped_vec_list.append(his_list[j])

        return grouped_vec_list, len(his_list)
    else:
        est_num_vec_each_block = len(his_list)/group_size
        base_num_vec_each_block = int(np.floor(len(his_list)/group_size))
        residual = est_num_vec_each_block - base_num_vec_each_block

        num_vec_has_extra_vec = int(np.round(residual * group_size))

        if residual == 0:
            for i in range(group_size):
                if len(his_list)<1:
                    print('len(his_list)<1')
                sum = np.zeros(len(his_list[0]))
                for j in range(base_num_vec_each_block):
                    if i*base_num_vec_each_block+j >= len(his_list):
                        print('i*num_vec_each_block+j')
                    sum += his_list[i*base_num_vec_each_block+j]
                grouped_vec_list.append(sum/base_num_vec_each_block)
        else:

            for i in range(group_size - num_vec_has_extra_vec):
                sum = np.zeros(len(his_list[0]))
                for j in range(base_num_vec_each_block):
                    if i*base_num_vec_each_block+j >= len(his_list):
                        print('i*base_num_vec_each_block+j')
                    sum += his_list[i*base_num_vec_each_block+j]
                    last_idx = i * base_num_vec_each_block + j
                grouped_vec_list.append(sum/base_num_vec_each_block)

            est_num = int(np.ceil(est_num_vec_each_block))
            start_group_idx = group_size - num_vec_has_extra_vec
            if len(his_list) - start_group_idx*base_num_vec_each_block >= est_num_vec_each_block:
                for i in range(start_group_idx,group_size):
                    sum = np.zeros(len(his_list[0]))
                    for j in range(est_num):
                        iidxx = last_idx + 1+(i-start_group_idx)*est_num+j
                        if  iidxx >= len(his_list) or iidxx<0:
                            print('last_idx + 1+(i-start_group_idx)*est_num+j')
                        sum += his_list[iidxx]
                    grouped_vec_list.append(sum/est_num)

        return grouped_vec_list, group_size

def temporal_decay_sum_history(data_set, key_set, output_size, group_size, within_decay_rate, group_decay_rate):
    sum_history = {}
    for key in key_set:
        vec_list = data_set[key] # basket list
        num_vec = len(vec_list) - 2
        his_list = []
        for idx in range(1,num_vec+1):
            his_vec = np.zeros(output_size)
            decayed_val = np.power(within_decay_rate, num_vec-idx)
            for ele in vec_list[idx]:
                his_vec[ele] = decayed_val
            his_list.append(his_vec)

        grouped_list, real_group_size = group_history_list(his_list, group_size)
        his_vec = np.zeros(output_size)
        for idx in range(real_group_size):
            decayed_val = np.power(group_decay_rate, group_size - 1 - idx)
            if idx>=len(grouped_list):
                print( 'idx: '+ str(idx))
                print('len(grouped_list): ' + str(len(grouped_list)))
            his_vec += grouped_list[idx]*decayed_val
        sum_history[key] = his_vec/real_group_size
    return sum_history

def KNN(query_set, target_set, k):
    history_mat = []
    for key in target_set.keys():
        history_mat.append(target_set[key])
    test_mat = []
    for key in query_set.keys():
        test_mat.append(query_set[key])
    # print('Finding k nearest neighbors...')
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(history_mat)
    distances, indices = nbrs.kneighbors(test_mat)
    # print('Finish KNN search.' )
    return indices, distances

def vec2label_list(pred_vec):
    # 转化为前100个label
    label_list = pred_vec.argsort()[::-1][:100].tolist()
    return label_list

def merge_history(sum_history_test, test_key_set, training_sum_history_test, training_key_set, index, alpha):
    merged_history = {}
    for test_key_id in range(len(test_key_set)):
        test_key = test_key_set[test_key_id]
        test_history = sum_history_test[test_key]
        sum_training_history = np.zeros(len(test_history))
        for indecis in index[test_key_id]:
            training_key = training_key_set[indecis]
            sum_training_history += training_sum_history_test[training_key]

        sum_training_history = sum_training_history/len(index[test_key_id])

        merge = test_history*alpha + sum_training_history*(1-alpha)
        merge = vec2label_list(merge) #transfer to label list
        merged_history[test_key] = merge

    return merged_history

def evaluate(data_history, training_key_set, test_key_set, input_size, group_size,
             within_decay_rate, group_decay_rate, num_nearest_neighbors, alpha):

    temporal_decay_sum_history_training = temporal_decay_sum_history(data_history,
                                                                     training_key_set, input_size,
                                                                     group_size, within_decay_rate,
                                                                     group_decay_rate) #don't need to change

    temporal_decay_sum_history_test = temporal_decay_sum_history(data_history,
                                                                 test_key_set, input_size,
                                                                 group_size, within_decay_rate,
                                                                 group_decay_rate)

    neighbour_index, distance = KNN(temporal_decay_sum_history_test, temporal_decay_sum_history_training,
                          num_nearest_neighbors)

    sum_history = merge_history(temporal_decay_sum_history_test, test_key_set, temporal_decay_sum_history_training,
                                training_key_set, neighbour_index, alpha)

    return sum_history

def main(argv):
    # param setting
    history_file = argv[1]
    future_file = argv[2]
    keyset_file = argv[3]
    num_nearest_neighbors = int(argv[4])
    within_decay_rate = float(argv[5])
    group_decay_rate = float(argv[6])
    alpha = float(argv[7])
    group_size = int(argv[8])
    topk = int(argv[9])

    with open(history_file, 'r') as f:
        data_history = json.load(f)
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)

    #vector size -> item num, get train meta test user list[]
    input_size = keyset['item_num']
    keyset_train = keyset['train']
    keyset_val = keyset['val']
    keyset_test = keyset['test']

    predicted_test = evaluate(data_history, keyset_train, keyset_test, input_size,
                                group_size, within_decay_rate, group_decay_rate,
                                num_nearest_neighbors, alpha)

    predicted_val = evaluate(data_history, keyset_train, keyset_val, input_size,
                                group_size, within_decay_rate, group_decay_rate,
                                num_nearest_neighbors, alpha)
    pred_dict = dict()
    pred_dict.update(predicted_val)
    pred_dict.update(predicted_test)

    print('Num. of top: ', topk)
    dataset_name = keyset_file.split('_')[0].split('/')[-1]
    keyset_ind = keyset_file.split('_')[-1].split('.')[0]
    pred_path = dataset_name + '_pred'+keyset_ind+'.json'
    with open(pred_path, 'w') as f:
        json.dump(pred_dict, f)


if __name__ == '__main__':
    main(sys.argv)
