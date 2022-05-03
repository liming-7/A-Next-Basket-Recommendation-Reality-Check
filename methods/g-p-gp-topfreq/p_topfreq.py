import pandas as pd
import json
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    parser.add_argument('--fold_id', type=int, default=0, help='x')
    args = parser.parse_args()
    dataset = args.dataset
    fold_id = args.fold_id
    data_history = pd.read_csv(f'../../dataset/{dataset}_history.csv')
    data_future = pd.read_csv(f'../../dataset/{dataset}_future.csv')

    pred_dict = dict()
    for user, user_data in data_future.groupby('user_id'):
        user_history = data_history[data_history['user_id'].isin([user])]
        history_items = user_history['product_id'].tolist()
        # print(history_items)
        s_pop_dict = dict()
        for item in history_items:
            if item not in s_pop_dict.keys():
                s_pop_dict[item] = 1
            else:
                s_pop_dict[item] += 1
        s_dict = sorted(s_pop_dict.items(), key=lambda d: d[1], reverse=True)
        pred = []
        for item, cnt in s_dict:
            pred.append(item)
        pred_dict[user] = pred
    
    if not os.path.exists('p_top_pred/'):
        os.makedirs('p_top_pred/')
    with open(f'p_top_pred/{dataset}_pred{fold_id}.json', 'w') as f:
        json.dump(pred_dict, f)



