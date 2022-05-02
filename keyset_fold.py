import pandas as pd
import json
import random
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    parser.add_argument('--fold_id', type=int, default=0, help='x')
    args = parser.parse_args()
    dataset = args.dataset
    fold_id = args.fold_id

    data_future = pd.read_csv(f'dataset/{dataset}_future.csv')
    data_history = pd.read_csv(f'dataset/{dataset}_history.csv')
    data = pd.concat([data_history, data_future])

    user = list(set(data_future['user_id']))
    user_num = len(user)
    random.shuffle(user)
    user = [str(user_id) for user_id in user]

    train_user = user[:int(user_num*4/5*0.9)]
    val_user = user[int(user_num*4/5*0.9):int(user_num*4/5)]
    test_user = user[int(user_num*4/5):]

    item_num = max(data['product_id'].tolist())+1
    keyset_dict = dict()
    keyset_dict['item_num'] = item_num
    keyset_dict['train'] = train_user
    keyset_dict['val'] = val_user
    keyset_dict['test'] = test_user

    print(keyset_dict)
    if not os.path.exists('keyset/'):
        os.makedirs('keyset/')
    keyset_file = f'keyset/{dataset}_keyset_{fold_id}.json'
    with open(keyset_file, 'w') as f:
        json.dump(keyset_dict, f)


