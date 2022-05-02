import numpy as np
import pandas as pd
# In order to deal with large sparse matrix we need to compresse them using
# the sparse sub_module of scipy lib
from scipy import sparse
import similaripy as sim

# User-Wise Popularity Method
def uwPopMat(df_train, n_items, recency=0):
  '''
    Calculate the user popularity matrix with the given recency window
    In: 
        df_train: train Dataframe
        n_items: #items
    Return : 
        User-wise Popularity matrix in csr sparse format 
  '''
  n_users = df_train.UID.unique().shape[0]
  if (recency>0):
    # Get the number of user baskets Bu 
    BUCount = df_train.groupby(['UID'])['order'].max().reset_index(name='Bu')
    # Calculate the denominator which equal to Min(recency,Bu) for each user
    BUCount['denominator'] = np.minimum(BUCount['Bu'],5)
    # Calculater the order index, form where we start counting item appearance in recent orders   
    BUCount['startindex'] = np.maximum(BUCount['Bu']-5,0)
    # Calcualte item appearance in recent orders   
    tmp = pd.merge(BUCount, df_train,on='UID')[['UID','PID','order','startindex']]
    tmp = tmp.loc[(tmp['order']>=tmp['startindex'])==True].groupby(['UID','PID'])['order'].count().reset_index(name='numerator')
    tmp = pd.merge(BUCount[['UID','denominator']],tmp,on='UID')
    # finally calculate the recency aware user-wise popularity
    tmp['Score'] = tmp['numerator']/tmp['denominator']
  else : 
    # Calculate user-wise popularity for each item
    BUCount = df_train.groupby(['UID'])['order'].max().reset_index(name='Bu')
    BUICount = df_train.groupby(['UID','PID'])['BID'].count().reset_index(name='Bui')
    tmp = pd.merge(BUICount, BUCount, on='UID')
    del BUICount
    tmp['Score'] = tmp['Bui']/tmp['Bu']
    del BUCount
    # get the 3 columns needed to construct our user-wise Popularity matrix
  df_UWpop =  tmp[['UID','PID','Score']]   
  del tmp
  # Generate user-wise popularity matrix in COOrdinate format
  UWP_mat = sparse.coo_matrix((df_UWpop.Score.values, (df_UWpop.UID.values, df_UWpop.PID.values)), shape=(n_users, n_items))
  del df_UWpop
  return sparse.csr_matrix(UWP_mat)

#2- Popularity-based Collaborative Filtering

#2.1- User Popularity-based CF (UP-CF)

def upcf(df_train, UWP_sparse, n_items, alpha = 0.25 ,q=5, k=10):
  n_users = df_train['UID'].unique().shape[0]
  df_user_item = df_train.groupby(['UID','PID']).size().reset_index(name="bool")[['UID','PID']]
  # Generate the User_item matrix using the parse matrix COOrdinate format.
  userItem_mat = sparse.coo_matrix((np.ones((df_user_item.shape[0])), (df_user_item.UID.values, df_user_item.PID.values)), shape=(n_users,n_items))
  # Calculate the asymmetric similarity cosine matrix 
  userSim = sim.asymmetric_cosine(sparse.csr_matrix(userItem_mat), alpha, k)
  # recommend k items to users
  user_recommendations = sim.dot_product(userSim.power(q), UWP_sparse, k)
  return user_recommendations

#2.2- Item popularity-based Collaborative Filtring 

def ipcf(df_train, UWP_sparse, n_items,alpha = 0.25, q=5, k=10):
  # Construct the item-basket sparse matrix 
  idMax_basket = df_train.BID.max()+1
  item_basket_mat = sparse.coo_matrix((np.ones((df_train.shape[0]),dtype=int), (df_train.PID.values, df_train.BID.values)), shape=(n_items,idMax_basket))
  # Convert it to Compressed Sparse Row format to exploit its efficiency in arithmetic operations 
  sparse_mat = sparse.csr_matrix(item_basket_mat)
  # Caculate the Asymetric Cosine Similarity matrix
  itemSimMat = sim.asymmetric_cosine(sparse_mat, None, alpha, k)
  # recommend k items to users
  UWP_sparse.shape, itemSimMat.shape
  user_recommendations = sim.dot_product(UWP_sparse, itemSimMat.power(q), k)
  return user_recommendations

