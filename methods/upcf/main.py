from util import instacartParser, prediction, evaluation
from NextBasketRecFramework import uwPopMat, upcf, ipcf
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # Path the dataset (for now we only applied it to the instacart dataset)
    parser.add_argument('--data_path', default='../data/instacart', help="")
    # Preprocessing args
    parser.add_argument('--item_threshold', default=10, type=int)
    parser.add_argument('--basket_threshold', default=2, type=int)
    parser.add_argument('--subdata', default=0.05, type=float)
    parser.add_argument('--verbose', default=True, type=bool)
    # Methods : {UWPop, UPCF, IPCF} with/out recency
    parser.add_argument('--method_name', default='UWPop')
    # Method's parameters
    parser.add_argument('--recency', default=0, type=int)
    parser.add_argument('--asymmetry', default=0, type=float)
    parser.add_argument('--locality', default=1, type=int)
    parser.add_argument('--top_k', default=5, type=int)
    
    args = parser.parse_args()
    # Get the prgm args
    data_path = args.data_path
    item_threshold, basket_threshold, subdata, verbose = args.item_threshold, args.basket_threshold, args.subdata, args.verbose
    method_name = args.method_name
    recency, alpha, q, k = args.recency, args.asymmetry, args.locality, args.top_k
    
    # Print framework parameters 
    print("method_name:",method_name)
    print("recency:",recency,'\nasymmetry:',alpha,"\nlocality:",q)
    print('=============')    
    
    # We return df_products for further uses like doing some analysis 
    df_train, dev_set, test_set, df_products = instacartParser(data_path, item_threshold, basket_threshold, subdata, verbose)
	
    # Show some dimensions
    print('=============')
    n_users = df_train.UID.unique().shape[0]
    n_items = df_products['PID'].unique().shape[0]
    n_baskets = df_train.BID.unique().shape[0]
    print('n_users:',n_users)
    print('n_items:',n_items)
    print('n_baskets:',n_baskets)
    print('=============')
    
    # Calculate the user wise popularity matrix 
    UWP_mat = uwPopMat(df_train, n_items, recency)
  
    # Calculate the user_recommendations matrix 
    if method_name == 'UWPop':
        user_recommendations = UWP_mat
    elif method_name == 'UPCF':
        user_recommendations = upcf(df_train, UWP_mat, n_items, alpha, q, k)
    elif method_name == 'IPCF':
        user_recommendations = ipcf(df_train, UWP_mat, n_items, alpha, q, k)

    print("User's Recommendations matrix dim:", user_recommendations.shape)
    # Predection
    score, pred = prediction(user_recommendations.toarray(), k)
    del user_recommendations
    
    # Evaluation
    test_ndcg, dev_ndcg = evaluation(score, pred, test_set, dev_set, k)
    print("Ndcg@",k,":")
    print("test score:",test_ndcg,"\ndev score:",dev_ndcg)
    del score,pred
