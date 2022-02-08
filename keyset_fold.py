import pandas as pd
import json
import random

#note the index
data_future = pd.read_csv('dataset/dunnhumby_future.csv')
data_history = pd.read_csv('dataset/dunnhumby_history.csv')
keyset_file = 'keyset/dunnhumby_keyset_0.json'
data = pd.concat([data_history, data_future])

user = list(set(data_future['user_id']))
user_num = len(user)
random.shuffle(user)
user = [str(user_id) for user_id in user]
print(user)

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
with open(keyset_file, 'w') as f:
    json.dump(keyset_dict, f)
# data_history = data_history[]



