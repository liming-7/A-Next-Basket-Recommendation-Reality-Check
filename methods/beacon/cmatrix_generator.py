import tensorflow as tf
import scipy.sparse as sp
import utils
import argparse
import json
from new_utils import *
# Model hyper-parameters
tf.flags.DEFINE_string("data_dir", 'dataset', "The input data directory (default: None)")
tf.flags.DEFINE_string("dataset", 'tafeng', "The input data directory (default: None)") # define name here
tf.flags.DEFINE_string("result_dir", 'temp_result', "The input data directory (default: None)")
tf.flags.DEFINE_integer("foldk", 0, "The input data directory (default: None)")
tf.flags.DEFINE_integer("nb_hop", 1, "The order of the real adjacency matrix (default:1)")


config = tf.flags.FLAGS
print("---------------------------------------------------")
print("Data_dir = " + str(config.data_dir))
print("\nParameters: " + str(config.__len__()))
for iterVal in config.__iter__():
    print(" + {}={}".format(iterVal, config.__getattr__(iterVal)))
print("Tensorflow version: ", tf.__version__)
print("---------------------------------------------------")

SEED_VALUES = [2, 9, 15, 44, 50, 55, 58, 79, 85, 92]

# ----------------------- MAIN PROGRAM -----------------------
data_dir = config.data_dir
temp_dir = config.result_dir
dataset = config.dataset
foldk = config.foldk
output_dir = f"temp_result/{dataset}_{foldk}_adj_matrix"

data_file = f'{data_dir}/{dataset}_merged.json'
keyset_path = f'../../keyset/{dataset}_keyset_{foldk}.json'
print("***************************************************************************************")
print("Output Dir: " + output_dir)

print("@Create output directory")
utils.create_folder(output_dir)

# Load train, validate & test
print("@Load train,validate&test data")
training_instances, train_uids = get_instances(data_file, keyset_path, mode='train')
# training_instances = utils.read_file_as_lines(training_file)
nb_train = len(training_instances)
print(" + Total training sequences: ", nb_train)

validate_instances, val_uids = get_instances(data_file, keyset_path, mode='val')
# validate_instances = utils.read_file_as_lines(validate_file)
nb_validate = len(validate_instances)
print(" + Total validating sequences: ", nb_validate)

test_instances, test_uids = get_instances(data_file, keyset_path, mode='test')
# validate_instances = utils.read_file_as_lines(validate_file)
nb_test = len(test_instances)
print(" + Total validating sequences: ", nb_test)

# Create dictionary
print("@Build knowledge")
MAX_SEQ_LENGTH, item_dict, reversed_item_dict, _ = build_knowledge(training_instances, validate_instances, test_instances)

print("#Statistic")
NB_ITEMS = len(item_dict)
print(" + Maximum sequence length: ", MAX_SEQ_LENGTH)
print(" + Total items: ", NB_ITEMS)

rmatrix_fpath = output_dir + "/r_matrix_" + str(config.nb_hop) + "w.npz"

print("@Build the real adjacency matrix")
real_adj_matrix = build_sparse_adjacency_matrix_v2(training_instances, validate_instances, item_dict)
real_adj_matrix = normalize_adj(real_adj_matrix)

mul = real_adj_matrix
with tf.device('/cpu:0'):
    w_mul = real_adj_matrix
    coeff = 1.0
    for w in range(1, config.nb_hop):
        coeff *= 0.85
        w_mul *= real_adj_matrix
        w_mul = remove_diag(w_mul)

        w_adj_matrix = normalize_adj(w_mul)
        mul += coeff * w_adj_matrix

    real_adj_matrix = mul

    sp.save_npz(rmatrix_fpath, real_adj_matrix)
    print(" + Save adj_matrix to" + rmatrix_fpath)
