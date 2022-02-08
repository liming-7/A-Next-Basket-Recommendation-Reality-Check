import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--same_embedding', type=int, default=1, help='whether use the same embedding in G1 and D')
parser.add_argument('--test_every_epoch', type=int, default=4, help='max_basket_num')
parser.add_argument('--pos_margin', type=float, default=0.1, help='max_basket_num')
parser.add_argument('--neg_margin', type=float, default=0.9, help='max_basket_num')

parser.add_argument('--neg_ratio', type=int, default=1, help='neg_ratio')
parser.add_argument('--device_id', type=int, default=0, help='GPU_ID')
parser.add_argument('--G1_flag', type=int, default=0, help='train_type :  with G1 1 / with no G1 -1 / from no G1 to G1 0')

parser.add_argument('--sd1', type=float, default=1, help='sd1')
parser.add_argument('--sd2', type=float, default=1, help='sd2')
parser.add_argument('--sd3', type=float, default=1, help='sd3')
parser.add_argument('--sd4', type=float, default=1, help='sd4')
parser.add_argument('--sd5', type=float, default=1, help='sd5')

parser.add_argument('--basket_pool_type', type=str, default='avg', help='basket_pool_type')
parser.add_argument('--num_layer', type=int, default=1, help='num_layer')
parser.add_argument('--test_type', type=int, default=3000, help='0:old 1000:1000 500:500')

parser.add_argument('--group_split1', type=int, default=4, help='basket_group_split')
parser.add_argument('--group_split2', type=int, default=6, help='basket_group_split')
parser.add_argument('--max_basket_size', type=int, default=50, help='max_basket_size')
parser.add_argument('--max_basket_num', type=int, default=50, help='max_basket_num')
parser.add_argument('--dataset', type=str, default='Instacart', help='dataset name')
parser.add_argument('--foldk', type=int, default=0, help='experiment fold')
parser.add_argument('--num_product', type=int, default=3920 , help='n_items TaFeng:9963 Instacart:8222 Delicious:6539')
parser.add_argument('--num_users', type=int, default=22530, help='n_users TaFeng:16060 Instacart:6885 Delicious:1735')
parser.add_argument('--distrisample', type=int, default= 0, help='')

parser.add_argument('--output_dir', type=str, default='./result', help='')
parser.add_argument('--log_fire', type=str, default='basemodel', help='basket_group_split') #_learning
parser.add_argument('--temp', type=float, default=1, help='') #1
parser.add_argument('--temp_min', type=float, default=0.3, help='') #0.3
parser.add_argument('--pretrain_epoch', type=int, default= 2, help='n_users TaFeng:16060 Instacart:6885 Delicious:1735')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
parser.add_argument('--epoch', type=int, default=50, help='the number of epochs to train for')
parser.add_argument('--ANNEAL_RATE', type=float, default=0.0001, help='ANNEAL_RATE') #0.0003
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--G1_lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=0.00001, help='l2 penalty')
parser.add_argument('--embedding_dim', type=int, default=64,help='hidden sise')
parser.add_argument('--alternative_train_epoch', type=int, default=5, help='max_basket_num')
parser.add_argument('--alternative_train_epoch_D', type=int, default=1, help='max_basket_num')
parser.add_argument('--alternative_train_batch', type=int, default=500, help='max_basket_num')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--history', type=int, default=1, help='history')
parser.add_argument('--temp_learn', type=int, default=1, help='temp_learn')
parser.add_argument('--before_epoch', type=int, default=0, help='basket_group_split') #18 63
parser.add_argument('--clip_value', type=float, default=0.1, help='dropout')
parser.add_argument('--to2', type=int, default= 0, help='n_users TaFeng:16060 Instacart:6885 Delicious:1735')

args = parser.parse_args()


# -*- coding:utf-8 -*-
class Config(object):
    def __init__(self):


        self.same_embedding = args.same_embedding
        self.neg_margin = args.neg_margin
        self.pos_margin = args.pos_margin

        self.alternative_train_epoch_D = args.alternative_train_epoch_D
        self.temp_learn = args.temp_learn
        self.output_dir = args.output_dir

        self.to2 = args.to2

        self.distrisample = args.distrisample
        self.pretrain_epoch = args.pretrain_epoch
        self.MODEL_DIR = f'models/{args.dataset}/'
        self.input_dir = 'dataset/{}'.format(args.dataset)+'_merged.json'
        self.keyset_dir = f'../../keyset/{args.dataset}_keyset_{args.foldk}.json'
        self.foldk = args.foldk
        self.dataset = args.dataset
        if args.dataset == "dunnhumby":
            self.num_users = 22530  #
            self.num_product = 3920
        if args.dataset == "tafeng":
            self.num_users = 13858
            self.num_product = 11997
        if args.dataset == "instacart":
            self.num_users = 19435
            self.num_product = 13897

        self.epochs = args.epoch
        self.device_id = args.device_id
        self.log_interval = 500  # num of batches between two logging #300


        self.item_list = list(range(args.num_product))
        self.test_ratio = 100  
        self.log_fire = args.log_fire
        self.test_type = args.test_type
        self.test_every_epoch = args.test_every_epoch
        self.clip = args.clip_value
      

        self.alternative_train_epoch = args.alternative_train_epoch
        self.alternative_train_batch = args.alternative_train_batch
        self.max_basket_size = args.max_basket_size 
        self.max_basket_num = args.max_basket_num 
        self.group_split1 = args.group_split1 
        self.group_split2 = args.group_split2
        self.batch_size = args.batch_size
        self.neg_ratio = args.neg_ratio 

        self.sd1 = args.sd1
        self.sd2 = args.sd2
        self.sd3 = args.sd3
        self.sd4 = args.sd4
        self.sd5 = args.sd5
        self.learning_rate = args.lr
        self.G1_lr = args.G1_lr
        self.dropout = args.dropout
        self.weight_decay = args.l2  
        self.basket_pool_type = args.basket_pool_type
        self.num_layer = args.num_layer 
   
        self.embedding_dim = args.embedding_dim  
        self.ANNEAL_RATE = args.ANNEAL_RATE

        self.temp = args.temp
        self.temp_min = args.temp_min
        self.before_epoch = args.before_epoch  

        self.G1_flag = args.G1_flag
        self.histroy = args.history  

    def list_all_member(self, logger):
        for name, value in vars(self).items():
            if not name.startswith('item'):
                logger.info('%s=%s' % (name, value))
