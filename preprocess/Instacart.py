import pandas as pd

user_order_d = pd.read_csv('../DataSource/instcart/orders.csv',
                         usecols=['user_id', 'order_number', 'order_id', 'eval_set'])
order_item_train = pd.read_csv('../DataSource/instcart/order_products__train.csv',
                               usecols=['order_id', 'product_id'])
order_item_prior = pd.read_csv('../DataSource/instcart/order_products__prior.csv',
                               usecols=['order_id', 'product_id'])
order_item = pd.concat([order_item_prior, order_item_train], ignore_index=True)

user_order = pd.merge(user_order_d, order_item, on='order_id', how='left')

user_order = user_order.dropna(how='any')

user_num = len(set(user_order['user_id'].tolist()))
user_num = int(user_num*0.1)
user_order = user_order[user_order['user_id'] <= user_num]

baskets = None
for user, user_data in user_order.groupby('user_id'):
    date_list = list(set(user_data['order_number'].tolist()))
    date_list = sorted(date_list)
    print(date_list)
    if len(date_list)>=3 and len(date_list)<=50:
        date_num = 1
        for date in date_list:
            date_data = user_data[user_data['order_number'].isin([date])]
            date_item = list(set(date_data['product_id'].tolist()))
            item_num = len(date_item)
            if baskets is None:
                baskets = pd.DataFrame({'user_id': pd.Series([user for i in range(item_num)]),
                                        'order_number': pd.Series([date_num for i in range(item_num)]),
                                        'product_id': pd.Series(date_item),
                                        'eval_set': pd.Series(['prior' for i in range(item_num)])})
                date_num += 1
            else:
                if date == date_list[-1]:#if date is the last. then add a tag here
                    temp = pd.DataFrame({'user_id': pd.Series([user for i in range(item_num)]),
                                            'order_number': pd.Series([date_num for i in range(item_num)]),
                                            'product_id': pd.Series(date_item),
                                            'eval_set': pd.Series(['train' for i in range(item_num)])})
                    date_num += 1
                    baskets = pd.concat([baskets, temp], ignore_index=True)
                else:
                    temp = pd.DataFrame({'user_id': pd.Series([user for i in range(item_num)]),
                                            'order_number': pd.Series([date_num for i in range(item_num)]),
                                            'product_id': pd.Series(date_item),
                                            'eval_set': pd.Series(['prior' for i in range(item_num)])})
                    date_num += 1
                    baskets = pd.concat([baskets, temp], ignore_index=True)


# print('Filter the data by seq length(user)')
# user_set = set()
# for user, user_data in baskets.groupby('user_id'):
#     basket_num = len(set(user_data['order_number']))
#     if basket_num>=3 and basket_num<=50:
#         user_set.add(user)
#
# baskets = baskets[baskets['user_id'].isin(user_set)].reset_index()
print('total transcations:', len(baskets))

item_set_all = set()
item_filter_dict = dict()
history_baskets = baskets[baskets['eval_set'].isin(['prior'])].reset_index()

for ind in range(len(history_baskets)):
    product_id = history_baskets['product_id'].iloc[ind]
    if product_id not in item_filter_dict:
        item_filter_dict[product_id] = 1
    else:
        item_filter_dict[product_id] += 1

for key in item_filter_dict.keys():
    if item_filter_dict[key]>=17:
        item_set_all.add(key)

print('Filter data use the training items.')
baskets = baskets[baskets['product_id'].isin(item_set_all)].reset_index()
print('After transcations:', len(baskets))

#### Filter by user
item_dict = dict()
item_ind = 1
user_dict = dict()
user_ind = 1
for ind in range(len(baskets)):
    product_id = baskets.at[ind, 'product_id']
    if product_id not in item_dict:
        item_dict[product_id] = item_ind
        item_ind += 1
    baskets.at[ind, 'product_id'] = item_dict[product_id]

    user_id = baskets.at[ind, 'user_id']
    if user_id not in user_dict:
        user_dict[user_id] = user_ind
        user_ind += 1
    baskets.at[ind, 'user_id'] = user_dict[user_id]
baskets = baskets.loc[:, ['user_id', 'order_number', 'product_id', 'eval_set']]
baskets.to_csv('dataset/instacart.csv', index=False)
