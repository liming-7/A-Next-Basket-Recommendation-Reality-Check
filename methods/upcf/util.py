import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def instacartParser(path, item_threshold=10, basket_threshold=2, subdata=0.1,verbose=True):
  '''
  IN:
    dataPath : os path to instacart data
    item_threshold : (default = 10 basket)
    basket_threshold : (default = 2 basket)
    subdata : (default: 10% of data)
    verbose: Boolean, (default:True)
  OUT:
    df_train : DataFrame where columns = [BID, UID, order,articles]
    dev_set, test_set : user-baskets items dict like {UID -> [PID,...], UID -> [PID,...], ...}
    df_products: DataFrame where columns = [PID, description, department, category]
  '''
  if verbose :
    start = time.time()
    # Start Time for calculating execution time
    print('Constructing products DataFrame object ...')

  # Read products.csv
  df_products= pd.read_csv(os.path.join(path,"products.csv"))
  df_products.columns = ['PID', 'description', 'categoryId', 'departmentId']
  # Read departments.csv and merge
  tmp = pd.read_csv(os.path.join(path,"departments.csv"))
  tmp.columns = ['departmentId', 'department']
  df_products = pd.merge(df_products, tmp, on='departmentId')
  # Read aisles.csv and merge 
  tmp = pd.read_csv(os.path.join(path,'aisles.csv'))
  tmp.columns = ['categoryId', 'category']
  df_products = pd.merge(df_products, tmp, on='categoryId')[['PID', 'description','department','category']]
  del tmp

  # preprocessing
  if verbose:
    print('Constructing order_products DataFrame object ...')

  df_order_products_prior = pd.read_csv(os.path.join(path,"order_products__prior.csv"))
  df_order_products_train = pd.read_csv(os.path.join(path,"order_products__train.csv"))
  df_order_products = pd.concat([df_order_products_prior, df_order_products_train])[['order_id', 'product_id']]
  df_order_products.columns= ['BID','PID']
  del df_order_products_prior, df_order_products_train,

  if verbose:
    print('Filtring items ...')
  # Remouving all items that appears in less than item_threshold baskets
  products_count = df_order_products['PID'].value_counts()
  df_order_products= df_order_products.loc[df_order_products['PID'].isin(products_count[products_count >= item_threshold].index)]
  del products_count
  # Updating production list 
  pd.merge(df_products,df_order_products['PID'],on='PID')

  if verbose:
    print('Reading Users order.csv file ...')
  df_orders = pd.read_csv(os.path.join(path,"orders.csv"))[['order_id', 'user_id', 'order_number', 'eval_set']]
  df_orders.columns = ['BID','UID','order', 'set']

  if verbose:
    print('Filtring Users...')
    print('Getting',subdata*100,'% of our dataset...')
  # User filtring
  # Remouving users with less than basket_threshold baskets
  user_count = df_orders['UID'].value_counts()
  user_filter = user_count[(user_count>=basket_threshold)&(np.random.rand(len(user_count))< subdata)]
  del user_count
  df_orders = df_orders[df_orders['UID'].isin(user_filter.index)]
  del user_filter

  # Reset UID 
  if verbose:
    print('Reset UID indexing')

  user_dict = dict(zip(df_orders['UID'].unique(),range(len(df_orders['UID'].unique()))))
  df_orders['UID'] = df_orders['UID'].map(user_dict)
  del user_dict
  # reset product index
  df_order_products = df_order_products.loc[df_order_products['BID'].isin(df_orders['BID'].unique())]
  df_products = df_products[df_products['PID'].isin(df_order_products['PID'].unique())]  
  product_dict = dict(zip(df_order_products['PID'].unique(),range(len(df_order_products['PID'].unique()))))
  df_products['PID'] = df_products['PID'].map(product_dict)
  df_order_products['PID'] = df_order_products['PID'].map(product_dict)
  del product_dict

  # Join Tables
  if verbose:
    print('Joining tables ...')
  df_data = pd.merge(df_orders, df_order_products, on= 'BID')
  del df_orders, df_order_products

  if verbose:
    print("spliting data ...")
  # Setting last baskets as dev/test sets
  last_basket_indexes = df_data.iloc[df_data.groupby(['UID'])['order'].idxmax()]['BID'].values
  df_data.loc[df_data['BID'].isin(last_basket_indexes),'set']='test'
  df_data.loc[df_data['set']=='prior', 'set'] = 'train'
  del last_basket_indexes

  # train test split data
  df_split = df_data[df_data['set']=='test'].groupby(by=['UID'])['PID'].apply(list).reset_index(name='articles')
  msk = (np.random.rand(len(df_split))<0.5)
  df_dev, df_test = df_split[msk], df_split[~msk]
  del df_split

  df_train = df_data[df_data['set']=='train'][['UID','BID','order','PID']]
  dev_set = dict(zip(df_dev['UID'],df_dev['articles']))
  test_set = dict(zip(df_test['UID'],df_test['articles']))

  del msk, df_dev, df_test
  # simple check
  assert (len(dev_set)+len(test_set))==(df_data['UID'].unique().shape[0])
  del df_data

  if verbose:
    print("processing took {0:.1f} sec".format(time.time() - start))
  
  return df_train, dev_set, test_set, df_products

def top_n(row, n):
    '''
    IN : 
      row : 1-D ndarray
      n   : int, number of top items
    OUT:
      top_values  : 1-D ndarray, Represent Top-n scores of the given row
      top_indices : 1-D ndarray, Represent Top-n users indices of the given row
    '''
    # Get user indices to sort the given row
    top_indices = row.argsort()[-n:][::-1]
    # Use the top_indices to get top_values score
    top_values  = row[top_indices]
    return top_values, top_indices

def prediction(predMat, k):
    '''
    In :
      predMat : the predection matrix 
      {UWPop, UB-CF, IB-CF }with/out recency (@r)
    Out :
      score, pred : ndarray of shape =(n_users, k)
      retun the top-k score and predection matrix
    '''
    n_users = predMat.shape[0]
    score = np.zeros((n_users, k))
    pred  = np.zeros((n_users, k))
    for i in range(n_users):
        score[i], pred[i] = top_n(predMat[i],k)
    return score.astype('float64'), pred.astype('int64')

def evaluation(score, pred, test_set, dev_set, k):
    '''
    Calculate the ndgs score for both test/dev sets for a giver user-item score and predection matrix.
    IN:
      score, pred: (n_user, m_items) ndarray matrix
      test_set, dev_set : user-baskets items dict like {UID -> [PID,...], UID -> [PID,...], ...}
      k :(default fixed to 5)    
    OUT:
      test_ndcg_score, dev_ndcg_score : type:int, evaluation metric for both test and dev set respectively  
    '''
    # Get the test and dev set User IDs
    test_keys = test_set.keys()
    dev_keys  = dev_set.keys()
    # Construct the True_relecvance and score vectors
    true_relevance_test = np.asarray([np.isin(pred[key],test_set[key]).astype(int) for key in test_keys])
    true_relevance_dev  = np.asarray([np.isin(pred[key],dev_set[key]).astype(int) for key in dev_keys])
    score_test = score[list(test_keys)]
    score_dev  = score[list(dev_keys)]
    # Calculate the ndgc@k evaluation metric 
    test_ndcg_score = ndcg_score(true_relevance_test, score_test, k=k)
    dev_ndcg_score  = ndcg_score(true_relevance_dev, score_dev, k=k)
    return test_ndcg_score, dev_ndcg_score
