import torch
import random
import numpy as np
import math
import time
import scipy.sparse as sp
from module.utils import *
from module.model_1 import NBModel
from module.config import Config
import pickle
import os.path
from module.logger import Logger
# from torch.utils.tensorboard import SummaryWriter


def load_dataset(fname, fuc):
    dataset = fuc()
    return dataset


# @profile
def train():

    if torch.cuda.is_available():
        torch.cuda.set_device(Config().device_id)

    isExist = os.path.exists(Config().output_dir)
    if not isExist:
        os.makedirs(Config().output_dir)

    logger_path = os.path.join(Config().output_dir, 'NB_{}_{}_{}.log'.format(Config().embedding_dim/2,Config().dataset, Config().log_fire))
    logger = Logger(logger_path)
    logger.info('*' * 150)
    logger.info(' update ndcg ')
    logger.info(' *************************************  model_1 *************************************** ')
    input_dir = os.path.join('./', Config().input_dir)
    keyset_dir = os.path.join('./', Config().keyset_dir)
    # why they use this? confusing
    # fuc = lambda x=None: get_dataset(input_dir, keyset_dir, Config().max_basket_size, Config().max_basket_num,
    #                                  Config().neg_ratio, Config().histroy)  # ,Config().histroy
    # dataset_path = os.path.join('./', 'dataset_{}_history_{}.pkl'.format(Config().dataset, Config().histroy))
    # dataset = load_dataset(dataset_path, fuc)


    TRAIN_DATASET, VALID_DATASET, TEST_DATASET, neg_sample, weights, itemnum, train_times, test_times, valid_times = get_dataset(input_dir, keyset_dir, Config().max_basket_size, Config().max_basket_num,
                                     Config().neg_ratio, Config().histroy)
    logger.info('train user nums : {}  test user nums : {}  valid user nums : {}'.format(train_times, test_times, valid_times))
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    Config().list_all_member(logger)
    NB = NBModel(Config(), device).to(device)

    def get_test_neg_set(test_type, batch_size, DATASET):
        neg_test_sample = dict()

        for batchid, (batch_userid, batch_input_seq, pad_batch_target_bsk, batch_history) in enumerate(
                get_batch_TEST_DATASET(DATASET, batch_size)):

            pad_batch_target_bsk = pad_batch_target_bsk.detach().cpu().numpy().tolist()  #
            for bid, pad_target_bsk in enumerate(pad_batch_target_bsk):  #
                uid_tensor = batch_userid[bid].to(device)
                uid = int(uid_tensor.detach().cpu().numpy())

                tar_b = list(set(pad_target_bsk) - set([-1]))
                if Config().histroy == 0:
                    neg_set = list(set(neg_sample[uid]) - set(tar_b))
                    S_pool = batch_input_seq[bid].cpu().numpy().tolist()
                    tar_b = list(set(tar_b) - set(S_pool))  #
                else:
                    neg_set = list(set(Config().item_list) - set(tar_b))

                if len(tar_b) < 1: continue

                len_t = len(tar_b)
                if test_type > 0:
                    neg_set = random.sample(neg_set, (test_type - len_t))
                else:
                    if len(neg_set) >= (Config().test_ratio * len(tar_b)):
                        neg_set = random.sample(neg_set, tar_b.__len__() * Config().test_ratio)
                    elif (Config().num_product) >= (Config().test_ratio * len(tar_b)):
                        neg_set = list(set(
                            random.sample(Config().item_list, tar_b.__len__() * Config().test_ratio)) - set(tar_b))
                    else:
                        neg_set = list(set(Config().item_list) - set(tar_b))  # -set(tar_b)
                neg_test_sample[uid] = neg_set
        return neg_test_sample

    test_neg_set = get_test_neg_set(Config().test_type, Config().batch_size, TEST_DATASET)
    valid_neg_set = get_test_neg_set(Config().test_type, Config().batch_size, VALID_DATASET)
    sd = Config().sd2

    def train_model(epoch, G1_flag, pretrain, temp=1, batch_size=256, alternative_train_batch=100, temp0 = Config().temp_min):
        NB.train()
        start_time = time.clock()
        loss_all = 0
        p1_all = 0
        p2_all = 0
        p3_all = 0
        p4_all = 0
        repeat_ratio_all = 0
        real_delete_all = 0
        sum_all = 0
        loss_all_count = 0
        temp = temp
        batch_num = 0
        for batchid, (batch_userid, batch_input_seq, batch_target, batch_history, batch_neg_items) in enumerate(
                get_batch_TRAIN_DATASET(TRAIN_DATASET, batch_size)):
            batch_num += 1
            if G1_flag == 0:
                temp = temp0 #Config().temp_min
            else:
                if batchid > 1:
                    if batchid % alternative_train_batch == 1:
                        temp = np.maximum(temp * np.exp(-Config().ANNEAL_RATE * batchid), Config().temp_min)
            batch_userid = batch_userid.to(device)
            batch_input_seq = batch_input_seq.to(device)
            batch_target = batch_target.to(device)
            batch_history = batch_history.to(device)
            # batch_neg_items = batch_neg_items.to(device)
            neg_set = []
            for u in batch_userid.detach().cpu().numpy().tolist():
                neg_items = random.sample(neg_sample[u], Config().neg_ratio)
                neg_set.append(neg_items)
            batch_neg_items = torch.tensor(neg_set, dtype=torch.long).to(device)

            # if batch_target.__len__() < 2: continue

            loss, _, (p1, p2, p3, p4), (real_rest_sum, real_rest_sum_int, all_sum, rest_ratio, repeat_ratio) = NB(temp, batch_userid,
                                                                                                     batch_input_seq,
                                                                                                     batch_target,
                                                                                                     weights,
                                                                                                     batch_history,
                                                                                                     batch_neg_items,
                                                                                                     train=True,
                                                                                                     G1flag=G1_flag,
                                                                                                     pretrain=pretrain,
                                                                                                     sd2=sd)

            optimizer_dict[G1_flag].zero_grad()
            loss.backward()

            optimizer_dict[G1_flag].step()

            loss_all += loss.data.item()
            real_delete_all += real_rest_sum_int.data.item()
            repeat_ratio_all += repeat_ratio.data.item()
            sum_all += all_sum.data.item()
            p1_all += p1.data.item()
            p2_all += p2.data.item()
            p3_all += p3.data.item()
            p4_all += p4.data.item()
            loss_all_count += 1

            if batchid % Config().log_interval == 0:
                elapsed = (time.clock() - start_time) / Config().log_interval
                cur_loss = loss.data.item()  # turn tensor into float
                cur_p1 = p1.data.item()
                cur_p2 = p2.data.item()
                cur_p3 = p3.data.item()
                cur_p4 = p4.data.item()
                start_time = time.clock()
                logger.info(
                    '[Training]| Epochs {:3d} | Batch {:5d}  | ms/batch {:02.2f} | Loss {:05.4f} | p1_Loss {:05.4f} | p2_Loss {:05.4f} | p3_Loss {:05.4f} | p4_Loss {:05.4f} | '
                        .format(epoch, batchid, elapsed, cur_loss, cur_p1, cur_p2, cur_p3, cur_p4))
                logger.info(
                    'real_rest_sum {:05.4f} | real_rest_sum_int {:05.4f} | all_sum {:05.4f} | rest_ratio {:05.4f} | repeat_ratio {:05.4f}'.format(
                        real_rest_sum.data.item(), real_rest_sum_int.item(), all_sum.data.item(),
                        rest_ratio.data.item(), repeat_ratio.data.item()))
        loss_all = loss_all / loss_all_count
        p1_all = p1_all / loss_all_count
        p2_all = p2_all / loss_all_count
        p3_all = p3_all / loss_all_count
        p4_all = p4_all / loss_all_count
        real_delete_ratio = real_delete_all / sum_all
        repeat_ratio_all = repeat_ratio_all/loss_all_count
        logger.info('batch_num: {}'.format(batch_num))
        logger.info(
            '[Training]| Epochs {:3d} | loss_all  {:05.4f} | p1_all  {:05.4f} | p2_all  {:05.4f} | p3_all  {:05.4f} | p4_all  {:05.4f} | real_rest_ratio {:05.4f} | repeat_ratio {:05.4f} |'.format(
                epoch, loss_all, p1_all, p2_all, p3_all, p4_all, real_delete_ratio, repeat_ratio_all))
        return loss_all, temp

    def valid_model_1000_top5(epoch, G1_flag, test_type=0, pretrain=0, temp=1, batch_size=256):

        NB.eval()

        hit_ratio_5 = 0
        recall_5 = 0
        precision_5 = 0
        f1_5 = 0
        ndcg_5 = 0
        mrr_5 = 0

        has_fake_user_5 = 0
        fake_length_5 = 0

        test_num = 0
        temp = temp

        with torch.no_grad():
            for batchid, (batch_userid, batch_input_seq, pad_batch_target_bsk, batch_history) in enumerate(
                    get_batch_TEST_DATASET(VALID_DATASET, batch_size)):
                if batchid % Config().alternative_train_batch == 1:
                    temp = np.maximum(temp * np.exp(-Config().ANNEAL_RATE * batchid), Config().temp_min)
                pad_batch_target_bsk = pad_batch_target_bsk.detach().cpu().numpy().tolist()  #
                for bid, pad_target_bsk in enumerate(pad_batch_target_bsk):  ##
                    uid_tensor = batch_userid[bid].to(device)
                    uid = int(uid_tensor.detach().cpu().numpy())

                    tar_b = list(set(pad_target_bsk) - set([-1]))  #
                    # if Config().histroy == 0:
                    #     S_pool = batch_input_seq[bid].cpu().numpy().tolist()
                    #     tar_b = list(set(tar_b) - set(S_pool))  #
                    if len(tar_b) < 1: continue

                    test_num += 1

                    input_tensor = batch_input_seq[bid].to(device)
                    history_tensor = batch_history[bid].to(device)

                    # len_t = len(tar_b)
                    # neg_set = random.sample(list(set(Config().item_list) - set(tar_b)), (test_type - len_t))
                    # neg_set = valid_neg_set[uid]

                    # minibatch_all_items = get_minibatch_split_all_items(neg_set)
                    # neg_set = tar_b + neg_set


                    target_items_tensor = torch.tensor(tar_b, dtype=torch.long).to(device)  # K
                    input_tensor_expand = input_tensor.view(1, -1).expand(len(tar_b), -1)  # K * B
                    uid_tensor_expand = uid_tensor.expand(len(tar_b))  # K
                    history_tensor_expand = history_tensor.view(1, -1).expand(len(tar_b), -1)  # K * n_items
                    rest_ratio, repeat_ratio, prob = NB(temp, uid_tensor_expand, input_tensor_expand, target_items_tensor, weights,
                                          history_tensor_expand,
                                          neg_set_tensor=None, train=False, G1flag=G1_flag, pretrain=pretrain, sd2=sd)
                    prob = prob.detach()[0, :].reshape(1, -1)  # 1*n_items
                    value_5, index_5 = torch.topk(prob.squeeze(), 20)
                    index_5 = index_5.tolist()
                    pred_bask = index_5 ## prediction is here!!

                    hit_len_5 = len(set(pred_bask) & set(tar_b))
                    hit_len_5 = len(set(pred_bask) & set(tar_b))

                    fake_length_5 += len(pred_bask)
                    if len(pred_bask) > 0:
                        has_fake_user_5 += 1

                    if hit_len_5 > 0:
                        ndcg_t, mrr_t = get_ndcg(pred_bask, tar_b)
                        ndcg_5 += ndcg_t
                        mrr_5 += mrr_t
                        hit_ratio_5 += 1
                        recall_5 += hit_len_5 / len(tar_b)
                        precision_5 += hit_len_5 / len(pred_bask)
                        f1_5 += (2 * (hit_len_5 / len(tar_b)) * (hit_len_5 / len(pred_bask)) / (
                                (hit_len_5 / len(tar_b)) + (hit_len_5 / len(pred_bask))))

                logger.info(1)

                avg_fake_basket_user_5 = fake_length_5 / has_fake_user_5

                hit_ratio_5 = hit_ratio_5 / test_num
                recall_5 = recall_5 / test_num
                precision_5 = precision_5 / test_num
                f1_5 = f1_5 / test_num
                ndcg_5 = ndcg_5 / test_num
                mrr_5 = mrr_5 / test_num

        #
        # logger.info(
        #     'valid_real_ratio_avg {:05.4f} valid_neg_ratio_avg {:05.4f} valid_repeat_ratio_avg {:05.4f} valid_neg_repeat_ratio_avg {:05.4f}'.format(test_real_ratio_avg,
        #                                                                          test_neg_ratio_avg,test_repeat_ratio_avg,test_neg_repeat_ratio_avg))

        logger.info(
            '[Validation] neg_sample TOP5 [Test]| Epochs {:3d} | Hit ratio {:02.4f} | recall {:05.4f} |  precision {:05.4f} | f1 {: 05.4f} | ndcg {: 05.4f} | mrr {: 05.4f} | have_fake_user {:3d} | avg_fake_length {: 05.4f} | all_valid_user_num  {:3d}'
                .format(epoch, hit_ratio_5, recall_5, precision_5, f1_5, ndcg_5, mrr_5, has_fake_user_5,
                        avg_fake_basket_user_5,
                        test_num))

        # logger.info('[Validation] p_n_score_differences_avg {:05.4f} '.format(p_n_score_differences_avg))
        logger.info('##############################################')
        return hit_ratio_5, recall_5, precision_5, f1_5, ndcg_5, mrr_5, avg_fake_basket_user_5

    def get_minibatch_split_all_items(item_list):
        minibatch_all_items = []
        minibatch_size = 500
        count = 0
        while count < len(item_list):
            if count + minibatch_size <= len(item_list):
                target_items = item_list[count:count + minibatch_size]
                count = count + minibatch_size
                minibatch_all_items.append(target_items)
            else:
                target_items = item_list[count:]
                count = len(item_list)
                minibatch_all_items.append(target_items)
        return minibatch_all_items

    def get_onehot_tensor(n_item, baskets):
        input_basket = []
        for basket in baskets:
            input_basket.append(np.squeeze(
                sp.coo_matrix(([1.] * basket.__len__(), ([0] * basket.__len__(), basket)),
                              shape=(1, n_item)).toarray()))
        input_basket = torch.tensor(input_basket, dtype=torch.float)
        return input_basket

    def get_tensor(seq_list):  #
        seq_tensor = get_onehot_tensor(Config().num_product, seq_list)
        return seq_tensor

    def get_ndcg(fake_basket, tar_b):
        u_dcg = 0
        u_idcg = 0
        rank_i = 0
        rank_flag = 0
        p_len = min(len(tar_b), 5)
        for k in range(5):  #
            if k < len(fake_basket):
                if fake_basket[k] in set(tar_b):  #
                    u_dcg += 1 / math.log(k + 1 + 1, 2)
                    if rank_flag == 0:
                        rank_i += 1 / (k + 1)  # min(p_len - 1, k)
                        rank_flag = 1

        idea = min(len(tar_b), 5)
        for k in range(idea):
            u_idcg += 1 / math.log(k + 1 + 1, 2)
        ndcg = u_dcg / u_idcg
        return ndcg, rank_i

    def test_model_1000_top5_new(epoch, G1_flag, group_split1=4, group_split2=6, test_type=0, pretrain=0, temp=1,
                                 batch_size=256):

        basket_length_dict = {}
        for i in [5]:
            basket_length_dict[i] = {}

            basket_length_dict[i]['<{}'.format(group_split1)] = {}
            basket_length_dict[i]['<{}'.format(group_split1)]['hit'] = []
            basket_length_dict[i]['<{}'.format(group_split1)]['recall'] = []
            basket_length_dict[i]['<{}'.format(group_split1)]['precision'] = []
            basket_length_dict[i]['<{}'.format(group_split1)]['f1'] = []
            basket_length_dict[i]['<{}'.format(group_split1)]['ndcg'] = []
            basket_length_dict[i]['<{}'.format(group_split1)]['mrr'] = []

            basket_length_dict[i]['{}<<{}'.format(group_split1, group_split2)] = {}
            basket_length_dict[i]['{}<<{}'.format(group_split1, group_split2)]['hit'] = []
            basket_length_dict[i]['{}<<{}'.format(group_split1, group_split2)]['recall'] = []
            basket_length_dict[i]['{}<<{}'.format(group_split1, group_split2)]['precision'] = []
            basket_length_dict[i]['{}<<{}'.format(group_split1, group_split2)]['f1'] = []
            basket_length_dict[i]['{}<<{}'.format(group_split1, group_split2)]['ndcg'] = []
            basket_length_dict[i]['{}<<{}'.format(group_split1, group_split2)]['mrr'] = []

            basket_length_dict[i]['>{}'.format(group_split2)] = {}
            basket_length_dict[i]['>{}'.format(group_split2)]['hit'] = []
            basket_length_dict[i]['>{}'.format(group_split2)]['recall'] = []
            basket_length_dict[i]['>{}'.format(group_split2)]['precision'] = []
            basket_length_dict[i]['>{}'.format(group_split2)]['f1'] = []
            basket_length_dict[i]['>{}'.format(group_split2)]['ndcg'] = []
            basket_length_dict[i]['>{}'.format(group_split2)]['mrr'] = []

        def get_index(prob, neg_set, tar_b):
            mask = get_tensor([neg_set]).to(device).view(1, -1).expand(prob.size(0), -1)  # n_items +1
            mask = (torch.ones_like(mask).to(device) - mask) * (-9999)
            prob = prob + mask  # K*n_items
            value_5, index_5 = torch.topk(prob, 5)

            tar_b_tensor = torch.tensor(tar_b).to(device)  #

            item_num = tar_b_tensor.size(0)  #
            index = torch.tensor(np.linspace(0, item_num, num=item_num, endpoint=False), dtype=torch.long)  # K
            pfake = prob[index, tar_b_tensor]  # K

            return value_5, index_5, pfake

        NB.eval()

        hit_ratio_5 = 0
        recall_5 = 0
        precision_5 = 0
        f1_5 = 0
        ndcg_5 = 0
        mrr_5 = 0

        time_count1 = 0

        has_fake_user_5 = 0
        fake_length_5 = 0

        test_num = 0
        temp = temp
        test_real_ratio = []
        test_neg_ratio = []

        test_repeat_ratio = []
        test_neg_repeat_ratio = []

        p_n_score_differences = []
        with torch.no_grad():
            for batchid, (batch_userid, batch_input_seq, pad_batch_target_bsk, batch_history) in enumerate(
                    get_batch_TEST_DATASET(TEST_DATASET, batch_size)):
                if batchid % Config().alternative_train_batch == 1:
                    temp = np.maximum(temp * np.exp(-Config().ANNEAL_RATE * batchid), Config().temp_min)
                pad_batch_target_bsk = pad_batch_target_bsk.detach().cpu().numpy().tolist()  #
                for bid, pad_target_bsk in enumerate(pad_batch_target_bsk):  ##
                    uid_tensor = batch_userid[bid].to(device)
                    uid = int(uid_tensor.detach().cpu().numpy())

                    tar_b = list(set(pad_target_bsk) - set([-1]))
                    if Config().histroy == 0:
                        S_pool = batch_input_seq[bid].cpu().numpy().tolist()
                        tar_b = list(set(tar_b) - set(S_pool))  #
                    if len(tar_b) < 1: continue

                    test_num += 1

                    input_tensor = batch_input_seq[bid].to(device)
                    history_tensor = batch_history[bid].to(device)

                    l = int(input_tensor.view(1, -1).view(1, -1, Config().max_basket_size).size(1))
                    mask_input = (input_tensor != -1).int()
                    avg_basket_size = float(mask_input.sum().data.item() / l)
                    if avg_basket_size < group_split1:
                        key_value = '<{}'.format(group_split1)
                    elif avg_basket_size > group_split2:
                        key_value = '>{}'.format(group_split2)
                    else:
                        key_value = '{}<<{}'.format(group_split1, group_split2)

                    len_t = len(tar_b)
                    neg_set = random.sample(list(set(Config().item_list) - set(tar_b)), (test_type - len_t))
                    # neg_set = test_neg_set[uid]

                    minibatch_all_items = get_minibatch_split_all_items(neg_set)
                    neg_set = tar_b + neg_set

                    target_items_tensor = torch.tensor(tar_b, dtype=torch.long).to(device)  # K
                    input_tensor_expand = input_tensor.view(1, -1).expand(len(tar_b), -1)  # K * B
                    uid_tensor_expand = uid_tensor.expand(len(tar_b))  # K
                    history_tensor_expand = history_tensor.view(1, -1).expand(len(tar_b), -1)  # K * n_items
                    rest_ratio,repeat_ratio, prob = NB(temp, uid_tensor_expand, input_tensor_expand, target_items_tensor, weights,
                                          history_tensor_expand,
                                          neg_set_tensor=None, train=False, G1flag=G1_flag, pretrain=pretrain, sd2=sd)
                    rest_ratio = rest_ratio.detach()
                    repeat_ratio = repeat_ratio.detach()
                    test_real_ratio.append(rest_ratio.data.item())
                    test_repeat_ratio.append(repeat_ratio.data.item())
                    t_value_5, t_index_5, pfake = get_index(prob, neg_set, tar_b)
                    value_5 = t_value_5
                    index_5 = t_index_5
                    pfake = pfake

                    for id, target_items in enumerate(minibatch_all_items):
                        target_items_tensor = torch.tensor(target_items, dtype=torch.long).to(device)  # K
                        input_tensor_expand = input_tensor.view(1, -1).expand(len(target_items), -1)  # K * B
                        uid_tensor_expand = uid_tensor.expand(len(target_items))  # K
                        history_tensor_expand = history_tensor.view(1, -1).expand(len(target_items), -1)  # K * n_items

                        rest_ratio, repeat_ratio,prob = NB(temp, uid_tensor_expand, input_tensor_expand, target_items_tensor,
                                              weights,
                                              history_tensor_expand,
                                              neg_set_tensor=None, train=False, G1flag=G1_flag, pretrain=pretrain,
                                              sd2=sd)
                        rest_ratio = rest_ratio.detach()
                        repeat_ratio = repeat_ratio.detach()
                        test_neg_ratio.append(rest_ratio.data.item())
                        test_neg_repeat_ratio.append(repeat_ratio.data.item())
                        t_value_5, t_index_5, t_pfake = get_index(prob, neg_set, target_items)
                        value_5 = torch.cat((value_5, t_value_5), dim=0)
                        index_5 = torch.cat((index_5, t_index_5), dim=0)
                        pfake = torch.cat((pfake, t_pfake), dim=0)

                    start_t1 = time.time()

                    p_pos_score = pfake[0:len(tar_b)]
                    p_neg_score = pfake[len(tar_b):]
                    # neg_top10_value, _ = torch.topk(p_neg_score, len(tar_b))
                    p_n_score_differences.append((torch.mean(p_pos_score) - torch.mean(p_neg_score)).data.item())

                    rank_pfake = pfake.cpu().numpy()
                    rank_pfake = -np.array(rank_pfake)
                    rank_index = np.argsort(rank_pfake)
                    select_index = rank_index[:5]
                    hit = np.array(neg_set)[select_index]
                    fake_basket_5 = list(hit)

                    hit_len_5 = len(set(fake_basket_5) & set(tar_b))
                    fake_length_5 += len(fake_basket_5)
                    if len(fake_basket_5) > 0:
                        has_fake_user_5 += 1

                    if hit_len_5 > 0:
                        ndcg_t, mrr_t = get_ndcg(fake_basket_5, tar_b)
                        ndcg_5 += ndcg_t
                        mrr_5 += mrr_t
                        hit_ratio_5 += 1
                        recall_5 += hit_len_5 / len(tar_b)
                        precision_5 += hit_len_5 / len(fake_basket_5)
                        f1_5 += (2 * (hit_len_5 / len(tar_b)) * (hit_len_5 / len(fake_basket_5)) / (
                                (hit_len_5 / len(tar_b)) + (hit_len_5 / len(fake_basket_5))))

                        basket_length_dict[5][key_value]['ndcg'].append(ndcg_t)
                        basket_length_dict[5][key_value]['mrr'].append(mrr_t)
                        basket_length_dict[5][key_value]['hit'].append(1)
                        basket_length_dict[5][key_value]['recall'].append(hit_len_5 / len(tar_b))
                        basket_length_dict[5][key_value]['precision'].append(hit_len_5 / len(fake_basket_5))
                        basket_length_dict[5][key_value]['f1'].append(
                            (2 * (hit_len_5 / len(tar_b)) * (hit_len_5 / len(fake_basket_5)) / (
                                    (hit_len_5 / len(tar_b)) + (hit_len_5 / len(fake_basket_5)))))
                    else:
                        basket_length_dict[5][key_value]['ndcg'].append(0)
                        basket_length_dict[5][key_value]['mrr'].append(0)
                        basket_length_dict[5][key_value]['hit'].append(0)
                        basket_length_dict[5][key_value]['recall'].append(0)
                        basket_length_dict[5][key_value]['precision'].append(0)
                        basket_length_dict[5][key_value]['f1'].append(0)

                    end_t1 = time.time()
                    time_count1 += (-start_t1 + end_t1)

        logger.info(1)
        test_loss = 0

        avg_fake_basket_user_5 = fake_length_5 / has_fake_user_5

        hit_ratio_5 = hit_ratio_5 / test_num
        recall_5 = recall_5 / test_num
        precision_5 = precision_5 / test_num
        f1_5 = f1_5 / test_num
        ndcg_5 = ndcg_5 / test_num
        mrr_5 = mrr_5 / test_num

        test_real_ratio_avg = np.mean(test_real_ratio)
        test_neg_ratio_avg = np.mean(test_neg_ratio)
        test_repeat_ratio_avg = np.nanmean(test_repeat_ratio)
        test_neg_repeat_ratio_avg = np.nanmean(test_neg_repeat_ratio)
        p_n_score_differences_avg = np.nanmean(p_n_score_differences)

        for kk in [5]:

            length_list = list(basket_length_dict[kk].keys())
            for length in length_list:
                hit_list = basket_length_dict[kk][length]['hit']
                recall_list = basket_length_dict[kk][length]['recall']
                precision_list = basket_length_dict[kk][length]['precision']
                f1_list = basket_length_dict[kk][length]['f1']
                ndcg_list = basket_length_dict[kk][length]['ndcg']
                mrr_list = basket_length_dict[kk][length]['mrr']
                logger.info(
                    'Epochs {:3d} topk {:3d}  basket_length {}  num {:3d}  Hit ratio {:02.4f} | recall {:05.4f} |  precision {:05.4f} | f1 {: 05.4f} | ndcg {: 05.4f} | mrr {: 05.4f} '.format(
                        epoch, kk, length, len(hit_list), np.mean(hit_list), np.mean(recall_list),
                        np.mean(precision_list), np.mean(f1_list), np.mean(ndcg_list), np.mean(mrr_list)
                    ))

            logger.info('...............................................')

        logger.info(
            'test_real_ratio_avg {:05.4f} test_neg_ratio_avg {:05.4f} test_repeat_ratio_avg {:05.4f} test_neg_repeat_ratio_avg {:05.4f}'.format(test_real_ratio_avg, test_neg_ratio_avg,test_repeat_ratio_avg,test_neg_repeat_ratio_avg))
        logger.info(
            'neg_sample TOP5 [Test]| Epochs {:3d} | Hit ratio {:02.4f} | recall {:05.4f} |  precision {:05.4f} | f1 {: 05.4f} | ndcg {: 05.4f} | mrr {: 05.4f} | have_fake_user {:3d} | avg_fake_length {: 05.4f} | all_test_user_num  {:3d}'
                .format(epoch, hit_ratio_5, recall_5, precision_5, f1_5, ndcg_5, mrr_5, has_fake_user_5,
                        avg_fake_basket_user_5,
                        test_num))
        logger.info('[Test] p_n_score_differences_avg {:05.4f} '.format(p_n_score_differences_avg))
        logger.info('##############################################')
        return hit_ratio_5, recall_5, precision_5, f1_5, ndcg_5, mrr_5, avg_fake_basket_user_5, test_loss

    def test_as_DREAM_new(epoch, G1_flag, group_split1=4, group_split2=6, test_type=0, pretrain=0, temp=1,
                          batch_size=256):

        basket_length_dict = {}
        for i in [5]:
            basket_length_dict[i] = {}

            basket_length_dict[i]['<{}'.format(group_split1)] = {}
            basket_length_dict[i]['<{}'.format(group_split1)]['hit'] = []
            basket_length_dict[i]['<{}'.format(group_split1)]['recall'] = []
            basket_length_dict[i]['<{}'.format(group_split1)]['precision'] = []
            basket_length_dict[i]['<{}'.format(group_split1)]['f1'] = []
            basket_length_dict[i]['<{}'.format(group_split1)]['ndcg'] = []
            basket_length_dict[i]['<{}'.format(group_split1)]['mrr'] = []

            basket_length_dict[i]['{}<<{}'.format(group_split1, group_split2)] = {}
            basket_length_dict[i]['{}<<{}'.format(group_split1, group_split2)]['hit'] = []
            basket_length_dict[i]['{}<<{}'.format(group_split1, group_split2)]['recall'] = []
            basket_length_dict[i]['{}<<{}'.format(group_split1, group_split2)]['precision'] = []
            basket_length_dict[i]['{}<<{}'.format(group_split1, group_split2)]['f1'] = []
            basket_length_dict[i]['{}<<{}'.format(group_split1, group_split2)]['ndcg'] = []
            basket_length_dict[i]['{}<<{}'.format(group_split1, group_split2)]['mrr'] = []

            basket_length_dict[i]['>{}'.format(group_split2)] = {}
            basket_length_dict[i]['>{}'.format(group_split2)]['hit'] = []
            basket_length_dict[i]['>{}'.format(group_split2)]['recall'] = []
            basket_length_dict[i]['>{}'.format(group_split2)]['precision'] = []
            basket_length_dict[i]['>{}'.format(group_split2)]['f1'] = []
            basket_length_dict[i]['>{}'.format(group_split2)]['ndcg'] = []
            basket_length_dict[i]['>{}'.format(group_split2)]['mrr'] = []

        NB.eval()

        hit_ratio_5 = 0
        recall_5 = 0
        precision_5 = 0
        f1_5 = 0
        ndcg_5 = 0
        mrr_5 = 0

        has_fake_user_5 = 0
        fake_length_5 = 0

        test_num = 0
        temp = temp
        with torch.no_grad():
            for batchid, (batch_userid, batch_input_seq, pad_batch_target_bsk, batch_history) in enumerate(
                    get_batch_TEST_DATASET(TEST_DATASET, batch_size)):
                if batchid % Config().alternative_train_batch == 1:
                    temp = np.maximum(temp * np.exp(-Config().ANNEAL_RATE * batchid), Config().temp_min)
                pad_batch_target_bsk = pad_batch_target_bsk.detach().cpu().numpy().tolist()  #
                for bid, pad_target_bsk in enumerate(pad_batch_target_bsk):  ##
                    uid_tensor = batch_userid[bid].to(device)
                    uid = int(uid_tensor.detach().cpu().numpy())

                    tar_b = list(set(pad_target_bsk) - set([-1]))
                    # if Config().histroy == 0:
                    #     S_pool = batch_input_seq[bid].cpu().numpy().tolist()
                    #     tar_b = list(set(tar_b) - set(S_pool))
                    if len(tar_b) < 1: continue

                    test_num += 1

                    input_tensor = batch_input_seq[bid].to(device)
                    history_tensor = batch_history[bid].to(device)

                    l = int(input_tensor.view(1, -1).view(1, -1, Config().max_basket_size).size(1))
                    mask_input = (input_tensor != -1).int()

                    # this is to eval over different bask size group!
                    avg_basket_size = float(mask_input.sum().data.item() / l)
                    if avg_basket_size < group_split1:
                        key_value = '<{}'.format(group_split1)
                    elif avg_basket_size > group_split2:
                        key_value = '>{}'.format(group_split2)
                    else:
                        key_value = '{}<<{}'.format(group_split1, group_split2)

                    neg_set = test_neg_set[uid]
                    neg_set = tar_b + neg_set

                    target_items_tensor = torch.tensor(tar_b, dtype=torch.long).to(device)  # K
                    input_tensor_expand = input_tensor.view(1, -1).expand(len(tar_b), -1)  # K * B
                    uid_tensor_expand = uid_tensor.expand(len(tar_b))  # K
                    history_tensor_expand = history_tensor.view(1, -1).expand(len(tar_b), -1)  # K * n_items
                    _,_, prob = NB(temp, uid_tensor_expand, input_tensor_expand, target_items_tensor, weights,
                                 history_tensor_expand,
                                 neg_set_tensor=None, train=False, G1flag=G1_flag, pretrain=pretrain, sd2=sd)
                    prob = prob.detach()[0, :].reshape(1, -1)  # 1*n_items
                    # mask = get_tensor([neg_set]).to(device).view(1, -1)
                    # mask = (torch.ones_like(mask).to(device) - mask) * (-9999)
                    prob = prob
                    value_5, index_5 = torch.topk(prob.squeeze(), 20)
                    index_5 = index_5.tolist()
                    fake_basket_5 = index_5 ## prediction is here!!

                    hit_len_5 = len(set(fake_basket_5) & set(tar_b))

                    fake_length_5 += len(fake_basket_5)
                    if len(fake_basket_5) > 0:
                        has_fake_user_5 += 1

                    if hit_len_5 > 0:
                        ndcg_t, mrr_t = get_ndcg(fake_basket_5, tar_b)
                        ndcg_5 += ndcg_t
                        mrr_5 += mrr_t
                        hit_ratio_5 += 1
                        recall_5 += hit_len_5 / len(tar_b)
                        precision_5 += hit_len_5 / len(fake_basket_5)
                        f1_5 += (2 * (hit_len_5 / len(tar_b)) * (hit_len_5 / len(fake_basket_5)) / (
                                (hit_len_5 / len(tar_b)) + (hit_len_5 / len(fake_basket_5))))

                        basket_length_dict[5][key_value]['ndcg'].append(ndcg_t)
                        basket_length_dict[5][key_value]['mrr'].append(mrr_t)
                        basket_length_dict[5][key_value]['hit'].append(1)
                        basket_length_dict[5][key_value]['recall'].append(hit_len_5 / len(tar_b))
                        basket_length_dict[5][key_value]['precision'].append(hit_len_5 / len(fake_basket_5))
                        basket_length_dict[5][key_value]['f1'].append(
                            (2 * (hit_len_5 / len(tar_b)) * (hit_len_5 / len(fake_basket_5)) / (
                                    (hit_len_5 / len(tar_b)) + (hit_len_5 / len(fake_basket_5)))))
                    else:
                        basket_length_dict[5][key_value]['ndcg'].append(0)
                        basket_length_dict[5][key_value]['mrr'].append(0)
                        basket_length_dict[5][key_value]['hit'].append(0)
                        basket_length_dict[5][key_value]['recall'].append(0)
                        basket_length_dict[5][key_value]['precision'].append(0)
                        basket_length_dict[5][key_value]['f1'].append(0)

        logger.info(1)
        test_loss = 0

        avg_fake_basket_user_5 = fake_length_5 / has_fake_user_5

        hit_ratio_5 = hit_ratio_5 / test_num
        recall_5 = recall_5 / test_num
        precision_5 = precision_5 / test_num
        f1_5 = f1_5 / test_num
        ndcg_5 = ndcg_5 / test_num
        mrr_5 = mrr_5 / test_num

        for kk in [5]:

            length_list = list(basket_length_dict[kk].keys())
            for length in length_list:
                hit_list = basket_length_dict[kk][length]['hit']
                recall_list = basket_length_dict[kk][length]['recall']
                precision_list = basket_length_dict[kk][length]['precision']
                f1_list = basket_length_dict[kk][length]['f1']
                ndcg_list = basket_length_dict[kk][length]['ndcg']
                mrr_list = basket_length_dict[kk][length]['mrr']
                logger.info(
                    'Epochs {:3d} topk {:3d}  basket_length {}  num {:3d}  Hit ratio {:02.4f} | recall {:05.4f} |  precision {:05.4f} | f1 {: 05.4f} | ndcg {: 05.4f} | mrr {: 05.4f} '.format(
                        epoch, kk, length, len(hit_list), np.mean(hit_list), np.mean(recall_list),
                        np.mean(precision_list), np.mean(f1_list), np.mean(ndcg_list), np.mean(mrr_list)
                    ))

            logger.info('...............................................')

        logger.info(
            'neg_sample TOP5 [Test]| Epochs {:3d} | Hit ratio {:02.4f} | recall {:05.4f} |  precision {:05.4f} | f1 {: 05.4f} | ndcg {: 05.4f} | mrr {: 05.4f}| have_fake_user {:3d} | avg_fake_length {: 05.4f} | all_test_user_num  {:3d}'
                .format(epoch, hit_ratio_5, recall_5, precision_5, f1_5, ndcg_5, mrr_5, has_fake_user_5,
                        avg_fake_basket_user_5,
                        test_num))

        logger.info('##############################################')
        return hit_ratio_5, recall_5, precision_5, f1_5, ndcg_5, mrr_5, avg_fake_basket_user_5, test_loss


    try:
        valid_hit_l = []
        valid_recall_l = []
        test_hit_l = []
        test_recall_l = []
        train_loss_l = []

        optimizer_dict = {}
        schedular_dict = {}
        optimizer_dict[2] = torch.optim.Adam([
            {'params': NB.G2.parameters()},
            {'params': NB.D.parameters()},
            {'params': NB.G0.parameters(), 'lr': Config().G1_lr}], lr=Config().learning_rate,
                                             weight_decay=Config().weight_decay)
        schedular_dict[2] = torch.optim.lr_scheduler.StepLR(optimizer_dict[2], step_size=3, gamma=1)

        optimizer_dict[1] = torch.optim.Adam([
            {'params': NB.G0.parameters()}]
            , lr=Config().G1_lr, weight_decay=Config().weight_decay)
        schedular_dict[1] = torch.optim.lr_scheduler.StepLR(optimizer_dict[1], step_size=3, gamma=1)

        optimizer_dict[0] = torch.optim.Adam([
            {'params': NB.G2.parameters()},
            {'params': NB.D.parameters()}]
            , lr=Config().learning_rate, weight_decay=Config().weight_decay)
        schedular_dict[0] = torch.optim.lr_scheduler.StepLR(optimizer_dict[0], step_size=3, gamma=1)

        best_hit_ratio = 0
        best_recall = 0
        best_precision = 0
        best_f1 = 0
        best_ndcg = 0
        best_mrr = 0
        temp = Config().temp
        if Config().G1_flag == 1: G1_flag = 2  ##
        if Config().G1_flag == 0: G1_flag = 0  ##
        if Config().G1_flag == -1: G1_flag = 0  ##
        pretrain = 0
        train_epoch = 0  ##
        pretrained_epoch = 0  ##

        logs = dict()

        first_batch_size = Config().batch_size
        al_batch = Config().alternative_train_batch
        B = first_batch_size

        if Config().before_epoch > 0:
            PATH = os.path.join(Config().MODEL_DIR, "base_model_{}_{}_{}.pt".format(Config().embedding_dim/2, Config().dataset,
                                                                                 'basemodel'))
            checkpoint0 = torch.load(PATH)
            checkpoint = checkpoint0['model_state_dict']
            model_dict = NB.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            NB.load_state_dict(model_dict)
            logger.info('reset from ... {}'.format(PATH))

            NB.G0.init_weight()

            test_as_DREAM_new(
                Config().before_epoch,
                G1_flag,
                Config().group_split1,
                Config().group_split2,
                test_type=Config().test_type,
                pretrain=pretrain,
                temp=Config().temp_min, batch_size=B)

            # test_model_1000_top5_new(
            #     Config().before_epoch,
            #     G1_flag,
            #     Config().group_split1,
            #     Config().group_split2,
            #     pretrain=pretrain,
            #     test_type=Config().test_type, temp=Config().temp_min, batch_size=B)

            if Config().before_epoch >= Config().pretrain_epoch:
                if Config().G1_flag == 0:
                    if pretrain == 0:
                        G1_flag = 1
                        pretrain = 1

        val_loss_before = dict()
        val_loss_before['hit'] = 0
        val_loss_before['recall'] = 0
        val_loss_before['precision'] = 0
        val_loss_before['f1'] = 0
        val_loss_before['ndcg'] = 0

        ############################################
        temp0 = Config().temp_min
        for epoch in range(Config().before_epoch + 1, Config().epochs + 1):
            sd = Config().sd2
            save_flag = 0
            if Config().G1_flag == 0:
                if pretrained_epoch == Config().pretrain_epoch:
                    if pretrain == 0:
                        G1_flag = 1
                        pretrain = 1

            if ((train_epoch % Config().alternative_train_epoch == 0) & (train_epoch > 0)):
                train_epoch = 0
                temp = Config().temp
                if Config().G1_flag == 0:
                    if G1_flag == 0:
                        G1_flag = 1
                    else:
                        train_epoch = Config().alternative_train_epoch - Config().alternative_train_epoch_D
                        save_flag = 1
                        G1_flag = 0
            temp0 = Config().temp_min # 默认为0
            logger.info(
                'G1_flag {} pretrain {} temp {} train_epoch {}  B {} al_batch {} lr {}  temp0 {}'.format(G1_flag, pretrain, temp,
                                                                                                train_epoch, B,
                                                                                                al_batch,
                                                                                                schedular_dict[
                                                                                                    G1_flag].get_lr()[
                                                                                                    0],temp0))
            ## train the model
            train_loss, temp0 = train_model(epoch, G1_flag, pretrain, temp=temp, batch_size=B,
                                            alternative_train_batch=al_batch, temp0 = temp0)

            ## validate model performance
            valid_hit_ratio, valid_recall, valid_precision, valid_f1, valid_ndcg, valid_mrr, valid_avg_fake_basket_user = valid_model_1000_top5(
                epoch, G1_flag,
                test_type=Config().test_type,
                pretrain=pretrain,
                temp=temp0,
                batch_size=B)

            valid_hit_l.append(valid_hit_ratio)
            valid_recall_l.append(valid_recall)
            train_loss_l.append(train_loss)

            learning_rate_scalar = schedular_dict[G1_flag].get_lr()[0]
            logs['learning_rate_G1flag_{}'.format(G1_flag)] = learning_rate_scalar
            logs['train_loss'] = train_loss
            logs['valid_hit'] = valid_hit_ratio
            logs['valid_recall'] = valid_recall
            logs['valid_pre'] = valid_precision
            logs['valid_f1'] = valid_f1
            logs['valid_ndcg'] = valid_ndcg

            schedular_dict[G1_flag].step()

            if (valid_hit_ratio > val_loss_before['hit']) | (valid_recall > val_loss_before['recall']) | (
                    valid_precision > val_loss_before['precision']) | (
                    valid_f1 > val_loss_before['f1']) | (valid_ndcg > val_loss_before['ndcg']):
                better = 0
                if valid_hit_ratio > val_loss_before['hit']: better += 1
                if valid_recall > val_loss_before['recall']: better += 1
                if valid_precision > val_loss_before['precision']: better += 1
                if valid_f1 > val_loss_before['f1']: better += 1
                if valid_ndcg > val_loss_before['ndcg']: better += 1
                if better > 1:
                    val_loss_before['hit'] = valid_hit_ratio
                    val_loss_before['recall'] = valid_recall
                    val_loss_before['precision'] = valid_precision
                    val_loss_before['f1'] = valid_f1
                    val_loss_before['ndcg'] = valid_ndcg
                    model_name = os.path.join(Config().MODEL_DIR,
                                              "base_model_{}_{}_{}.pt".format(Config().embedding_dim / 2,
                                                                              Config().dataset,
                                                                              Config().log_fire))
                    checkpoint = {'epoch': epoch,
                                  'model_state_dict': NB.state_dict(),
                                  }
                    torch.save(checkpoint, model_name)
                    logger.info("Save model as %s" % model_name)
            #
            # if ((((train_epoch + 1) % Config().alternative_train_epoch == 0)) | (
            #         epoch % Config().test_every_epoch == 0) | (save_flag == 1) | (epoch == Config().epochs) | (
            #         pretrained_epoch == (Config().pretrain_epoch - 1))):
            #
            #     if ((G1_flag == 0) & (pretrain == 0)):
            #         hit_ratio, recall, precision, f1, ndcg, mrr, avg_fake_basket_user, test_loss = test_as_DREAM_new(
            #             epoch,
            #             G1_flag,
            #             Config().group_split1,
            #             Config().group_split2,
            #             test_type=Config().test_type,
            #             pretrain=pretrain,
            #             temp=temp, batch_size=B)
            #         logs['hit'] = hit_ratio
            #         logs['recall'] = recall
            #         logs['precision'] = precision
            #         logs['f1'] = f1
            #         logs['ndcg'] = ndcg
            #
            #         if (hit_ratio > best_hit_ratio) | (f1 > best_f1) | (ndcg > best_ndcg):
            #             better = 0
            #             if hit_ratio > best_hit_ratio: better += 1
            #             if f1 > best_f1: better += 1
            #             if ndcg > best_ndcg: better += 1
            #             if better > 1:
            #                 model_name = os.path.join(Config().MODEL_DIR, "base_model_{}_{}_{}.pt".format(Config().embedding_dim/2,Config().dataset,
            #                                                                          Config().log_fire))
            #                 checkpoint = {'epoch': epoch,
            #                               'model_state_dict': NB.state_dict(),
            #                               }
            #                 torch.save(checkpoint, model_name)
            #                 logger.info("Save model as %s" % model_name)
            #     else:
            #         hit_ratio, recall, precision, f1, ndcg, mrr, avg_fake_basket_user, test_loss = test_model_1000_top5_new(
            #             epoch,
            #             G1_flag,
            #             Config().group_split1,
            #             Config().group_split2,
            #             pretrain=pretrain,
            #             test_type=Config().test_type, temp=temp0, batch_size=B)
            #         logs['hit'] = hit_ratio
            #         logs['recall'] = recall
            #         logs['precision'] = precision
            #         logs['f1'] = f1
            #         logs['ndcg'] = ndcg
            #
            #     test_hit_l.append(hit_ratio)
            #     test_recall_l.append(recall)
            #
            #     if (hit_ratio > best_hit_ratio) | (recall > best_recall) | (precision > best_precision) | (
            #             f1 > best_f1) | (ndcg > best_ndcg) | (mrr > best_mrr):
            #         better = 0
            #         if hit_ratio > best_hit_ratio: better += 1
            #         if recall > best_recall: better += 1
            #         if precision > best_precision: better += 1
            #         if f1 > best_f1: better += 1
            #         if ndcg > best_ndcg: better += 1
            #         if better > 1:
            #             best_PATH = os.path.join(Config().MODEL_DIR,
            #                                      "model_1_{}_{}_{}.pt".format(Config().embedding_dim/2,Config().dataset,
            #                                                                                    Config().log_fire))
            #             checkpoint = {'epoch': epoch,
            #                           'temp0': temp0,
            #                           'G1_flag':G1_flag,
            #                           'model_state_dict': NB.state_dict(),
            #                           }
            #             torch.save(checkpoint, best_PATH)
            #             logger.info("Save model as %s" % best_PATH)
            #
            #         best_hit_ratio = max(hit_ratio, best_hit_ratio)
            #         best_recall = max(recall, best_recall)
            #         best_precision = max(precision, best_precision)
            #         best_f1 = max(f1, best_f1)
            #         best_ndcg = max(ndcg, best_ndcg)
            #         best_mrr = max(mrr, best_mrr)
            #         logs['besthit'] = best_hit_ratio
            #         logs['bestrecall'] = best_recall
            #         logs['bestprecision'] = best_precision
            #         logs['bestf1'] = best_f1
            #         logs['bestndcg'] = best_ndcg
            #
            #     logger.info(
            #         'Epochs {:3d} best_hit {:05.4f} best_recall {:05.4f} best_precision {:05.4f} best_f1 {:05.4f} best_ndcg {:05.4f}  best_mrr {:05.4f} '.format(
            #             epoch, best_hit_ratio, best_recall, best_precision, best_f1, best_ndcg, best_mrr))
            #
            #     logger.info(
            #         'Epochs {:3d} loss_all {:05.4f} Hit ratio {:02.4f} recall {:05.4f} precision {:05.4f} f1 {: 05.4f} ndcg {: 05.4f} mrr {: 05.4f} avg_fake_length {:05.4f}'
            #             .format(epoch, train_loss, hit_ratio, recall, precision, f1, ndcg, mrr,
            #                     avg_fake_basket_user))

            # for key, value in logs.items():
            #     tb_writer.add_scalar(key, value, epoch)

            temp = Config().temp
            if Config().G1_flag == 0:
                train_epoch += 1
            if pretrain == 0:
                if G1_flag == 0:
                    pretrained_epoch += 1
                    train_epoch = 0
        logger.info('valid_hit {}'.format(valid_hit_l))
        logger.info('valid_recall {}'.format(valid_recall_l))
        logger.info('test_hit {}'.format(test_hit_l))
        logger.info('test_hit {}'.format(test_recall_l))
        logger.info('train_loss {}'.format(train_loss_l))
    except KeyboardInterrupt:
        logger.info('Early Stopping!')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train()
