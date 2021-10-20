from baselines.metrics import *
import pandas as pd
import json


def get_exp_recall(dataname, k, ind):
    history_file = '../dataset/'+dataname+'_history.csv'
    keyset_file = '../keyset/'+dataname+'_keyset_'+str(ind)+'.json'
    pred_file = 'tifuknn/'+dataname+'_pred'+str(ind)+'.json'
    truth_file = '../jsondata/'+dataname+'_future.json'
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)
    with open(pred_file, 'r') as f:
        data_pred = json.load(f)
    with open(truth_file, 'r') as f:
        data_truth = json.load(f)

    data_history = pd.read_csv(history_file)

    ndcg_repeat = []
    recall_repeat = []
    ndcg_explore = []
    recall_explore = []
    phr = []

    for user in keyset['test']:
        pred = data_pred[user]
        truth = data_truth[user][1]
        # print(user)
        user_history = data_history[data_history['user_id'].isin([int(user)])]
        repeat_items = list(set(user_history['product_id']))
        # print('user:', user)
        truth_repeat = list(set(truth)&set(repeat_items)) #might be none
        truth_explore = list(set(truth)-set(truth_repeat)) #might be none
        pred_explore = list(set(pred)-set(repeat_items))
        pred_repeat = list(set(pred)&set(repeat_items))

        if len(truth_repeat)>0:
            pred_repeat = pred_repeat[:len(truth_repeat)]
            u_recall_repeat = get_Recall(truth_repeat, pred_repeat, len(truth_repeat))# here repeat truth, since repeat items might not in the groundtruth
            recall_repeat.append(u_recall_repeat)

            u_ndcg_repeat = get_NDCG(truth_repeat, pred_repeat, len(truth_repeat))
            ndcg_repeat.append(u_ndcg_repeat)

        if len(truth_explore)>0:
            pred_explore = pred_explore[:len(truth_explore)]
            u_recall_explore = get_Recall(truth_explore, pred_explore, len(truth_explore))
            u_ndcg_explore = get_NDCG(truth_explore, pred_explore, len(truth_explore))
            recall_explore.append(u_recall_explore)
            ndcg_explore.append(u_ndcg_explore)

    return np.mean(recall_repeat), np.mean(ndcg_repeat), np.mean(recall_explore), np.mean(ndcg_explore)

for name in ['tafeng', 'dunnhumby', 'instacart']:
    print(name)

    for k in [10, 20]:
        recall_rep = []
        ndcg_rep = []
        recall_exp = []
        ndcg_exp = []
        for ind in [0, 1, 2]:
            rep_r, rep_n, exp_r, exp_n = get_exp_recall(name, k, ind)
            recall_rep.append(rep_r)
            recall_exp.append(exp_r)
            ndcg_rep.append(rep_n)
            ndcg_exp.append(exp_n)
        print(k)
        print('repeat:', np.mean(recall_rep), np.mean(ndcg_rep))
        print('explore', np.mean(recall_exp), np.mean(ndcg_exp))
