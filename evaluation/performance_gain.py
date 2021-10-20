from baselines.metrics import *
import pandas as pd
import json


def get_exp_recall(dataname, k, ind):
    history_file = '../dataset/'+dataname+'_history.csv'
    keyset_file = '../keyset/'+dataname+'_keyset_'+str(ind)+'.json'
    pred_file = 'RACF/pred/'+dataname+'_pred'+str(ind)+'.json'
    # pred_file = 'gp-topfreq/pred/'+dataname+'_pred.json'
    truth_file = '../jsondata/'+dataname+'_future.json'
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)
    with open(pred_file, 'r') as f:
        data_pred = json.load(f)
    with open(truth_file, 'r') as f:
        data_truth = json.load(f)

    data_history = pd.read_csv(history_file)

    expl_ndcg = []
    expl_recall = []
    expl_phr = []

    rep_ndcg = []
    rep_recall = []
    rep_phr = []

    for user in keyset['test']:
        pred = data_pred[user][:k]
        truth = data_truth[user][1]
        # print(user)
        user_history = data_history[data_history['user_id'].isin([int(user)])]
        repeat_items = list(set(user_history['product_id']))
        # print('user:', user)
        pred_rep = list(set(pred)&set(repeat_items))
        r_ndcg = get_NDCG(truth, pred_rep, k)
        rep_ndcg.append(r_ndcg)
        r_recall = get_Recall(truth, pred_rep, k)
        rep_recall.append(r_recall)
        r_hit = get_HT(truth, pred_rep, k)
        rep_phr.append(r_hit)

        pred_explore = list(set(pred)-set(pred_rep))
        e_ndcg = get_NDCG(truth, pred_explore, k)
        expl_ndcg.append(e_ndcg)
        e_recall = get_Recall(truth, pred_explore, k)
        expl_recall.append(e_recall)
        e_hit = get_HT(truth, pred_explore, k)
        expl_phr.append(e_hit)

    return np.mean(rep_recall), np.mean(expl_recall)

for name in ['tafeng', 'dunnhumby', 'instacart']:
    print(name)

    for k in [10, 20]:
        rep_recall = []
        expl_recall = []
        for ind in [0, 1, 2]:
            ind_r_recall, ind_e_recall = get_exp_recall(name, k, ind)
            rep_recall.append(ind_r_recall)
            expl_recall.append(ind_e_recall)
        print(k)
        print(np.mean(rep_recall), np.mean(expl_recall))
