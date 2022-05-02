import json
import sys
import os
from tqdm import tqdm
from utils.metric import evaluate
from utils.data_container import get_data_loader
from utils.load_config import get_attribute
from utils.util import convert_to_gpu
from train.train_main import create_model
from utils.util import load_model

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    parser.add_argument('--fold_id', type=int, default=0, help='x')
    parser.add_argument('--best_model_path', type=str, required=True)
    args = parser.parse_args()
    dataset = args.dataset
    fold = args.fold_id
    model_path = parser.best_model_path
    history_path = f'../../jsondata/{dataset}_history.json'
    future_path = f'../../jsondata/{dataset}_future.json'
    keyset_path = f'../../keyset/{dataset}_keyset_{fold}.json'
    
    pred_path = f'{dataset}_pred{fold}.json'
    truth_path = f'{dataset}_truth{fold}.json'
    with open(keyset_path, 'r') as f:
        keyset = json.load(f)

    model = create_model()
    model = load_model(model, model_path)

    data_loader = get_data_loader(history_path=history_path,
                                        future_path=future_path,
                                    keyset_path=keyset_path,
                                    data_type='test',
                                    batch_size=1,
                                    item_embedding_matrix=model.item_embedding)

    model.eval()

    pred_dict = dict()
    truth_dict = dict()
    test_key = keyset['test']
    user_ind = 0
    for step, (g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency) in enumerate(
                    tqdm(data_loader)):
        pred_data = model(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)
        pred_list = pred_data.detach().squeeze(0).numpy().argsort()[::-1][:100].tolist()
        truth_list = truth_data.detach().squeeze(0).numpy().argsort()[::-1][:100].tolist()
        pred_dict[test_key[user_ind]] = pred_list
        truth_dict[test_key[user_ind]] = truth_list
        user_ind += 1

    with open(pred_path, 'w') as f:
        json.dump(pred_dict, f)
    with open(truth_path, 'w') as f:
        json.dump(truth_dict, f)

