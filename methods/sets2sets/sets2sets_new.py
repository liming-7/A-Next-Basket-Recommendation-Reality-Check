import numpy as np
import random
import sys
import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

num_iter = 20 #epoch number
hidden_size = 32
num_layers = 1

# only one can be set 1
use_embedding = 1
use_linear_reduction = 0

atten_decoder = 1
use_dropout = 0
use_average_embedding = 1

labmda = 10

MAX_LENGTH = 100
learning_rate = 0.001
print_val = 3000
use_cuda = torch.cuda.is_available()


class EncoderRNN_new(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN_new, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reduction = nn.Linear(input_size, hidden_size)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.time_embedding = nn.Embedding(input_size, hidden_size)
        self.time_weight = nn.Linear(input_size, input_size)
        if use_embedding or use_linear_reduction:
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        else:
            self.gru = nn.GRU(input_size, hidden_size, num_layers)

    def forward(self, input, hidden):
        if use_embedding:
            list = Variable(torch.LongTensor(input).view(-1, 1))
            if use_cuda:
                list = list.cuda()
            average_embedding = Variable(torch.zeros(hidden_size)).view(1, 1, -1)
            vectorized_input = Variable(torch.zeros(self.input_size)).view(-1)
            if use_cuda:
                average_embedding = average_embedding.cuda()
                vectorized_input = vectorized_input.cuda()

            for ele in list:
                embedded = self.embedding(ele).view(1, 1, -1)
                tmp = average_embedding.clone()
                average_embedding = tmp + embedded
                vectorized_input[ele] = 1

            if use_average_embedding:
                tmp = [1] * hidden_size
                length = Variable(torch.FloatTensor(tmp))
                if use_cuda:
                    length = length.cuda()
                # for idx in range(hidden_size):
                real_ave = average_embedding.view(-1) / length
                average_embedding = real_ave.view(1, 1, -1)

            embedding = average_embedding
        else:
            tensorized_input = torch.from_numpy(input).clone().type(torch.FloatTensor)
            inputs = Variable(torch.unsqueeze(tensorized_input, 0).view(1, -1))
            if use_cuda:
                inputs = inputs.cuda()
            if use_linear_reduction == 1:
                reduced_input = self.reduction(inputs)
            else:
                reduced_input = inputs

            embedding = torch.unsqueeze(reduced_input, 0)

        output, hidden = self.gru(embedding, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(num_layers, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN_new(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout_p=0.2, max_length=MAX_LENGTH):
        super(AttnDecoderRNN_new, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_embedding or use_linear_reduction:
            self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
            self.attn1 = nn.Linear(self.hidden_size + output_size, self.hidden_size)
        else:
            self.attn = nn.Linear(self.hidden_size + self.output_size, self.output_size)

        if use_embedding or use_linear_reduction:
            self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.attn_combine3 = nn.Linear(self.hidden_size * 2 + output_size, self.hidden_size)
        else:
            self.attn_combine = nn.Linear(self.hidden_size + self.output_size, self.hidden_size)
        self.attn_combine5 = nn.Linear(self.output_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.reduction = nn.Linear(self.output_size, self.hidden_size)
        if use_embedding or use_linear_reduction:
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        else:
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, history_record, last_hidden):
        if use_embedding:
            list = Variable(torch.LongTensor(input).view(-1, 1))
            if use_cuda:
                list = list.cuda()
            average_embedding = Variable(torch.zeros(hidden_size)).view(1, 1, -1)
            if use_cuda:
                average_embedding = average_embedding.cuda()

            for ele in list:
                embedded = self.embedding(ele).view(1, 1, -1)
                tmp = average_embedding.clone()
                average_embedding = tmp + embedded

            if use_average_embedding:
                tmp = [1] * hidden_size
                length = Variable(torch.FloatTensor(tmp))
                if use_cuda:
                    length = length.cuda()
                # for idx in range(hidden_size):
                real_ave = average_embedding.view(-1) / length
                average_embedding = real_ave.view(1, 1, -1)

            embedding = average_embedding
        else:
            tensorized_input = torch.from_numpy(input).clone().type(torch.FloatTensor)
            inputs = Variable(torch.unsqueeze(tensorized_input, 0).view(1, -1))
            if use_cuda:
                inputs = inputs.cuda()
            if use_linear_reduction == 1:
                reduced_input = self.reduction(inputs)
            else:
                reduced_input = inputs

            embedding = torch.unsqueeze(reduced_input, 0)

        if use_dropout:
            droped_ave_embedded = self.dropout(embedding)
        else:
            droped_ave_embedded = embedding

        history_context = Variable(torch.FloatTensor(history_record).view(1, -1))
        if use_cuda:
            history_context = history_context.cuda()

        attn_weights = F.softmax(
            self.attn(torch.cat((droped_ave_embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        element_attn_weights = F.softmax(
            self.attn1(torch.cat((history_context, hidden[0]), 1)), dim=1)

        output = torch.cat((droped_ave_embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        linear_output = self.out(output[0])

        value = torch.sigmoid(self.attn_combine5(history_context).unsqueeze(0))

        one_vec = Variable(torch.ones(self.output_size).view(1, -1))
        if use_cuda:
            one_vec = one_vec.cuda()

        res = history_context.clone()
        res[history_context != 0] = 1
        # Linear后，要mask掉其他位置的，value为weight，gru的output需要减去weight，再weight*history_context.
        output = F.softmax(linear_output * (one_vec - res * value[0]) + history_context * value[0], dim=1)

        return output.view(1, -1), hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(num_layers, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class custom_MultiLabelLoss_torch(nn.modules.loss._Loss):
    def __init__(self):
        super(custom_MultiLabelLoss_torch, self).__init__()

    def forward(self, pred, target, weights):
        #balance the mseloss, incase that some items occurs frequently in the training dataset
        mseloss = torch.sum(weights * torch.pow((pred - target), 2))
        pred = pred.data
        target = target.data

        ones_idx_set = (target == 1).nonzero()
        zeros_idx_set = (target == 0).nonzero()

        ones_set = torch.index_select(pred, 1, ones_idx_set[:, 1])
        zeros_set = torch.index_select(pred, 1, zeros_idx_set[:, 1])

        repeat_ones = ones_set.repeat(1, zeros_set.shape[1])
        repeat_zeros_set = torch.transpose(zeros_set.repeat(ones_set.shape[1], 1), 0, 1).clone()
        repeat_zeros = repeat_zeros_set.reshape(1, -1)
        difference_val = -(repeat_ones - repeat_zeros)
        exp_val = torch.exp(difference_val)
        exp_loss = torch.sum(exp_val)
        normalized_loss = exp_loss / (zeros_set.shape[1] * ones_set.shape[1])
        set_loss = Variable(torch.FloatTensor([labmda * normalized_loss]), requires_grad=True)
        if use_cuda:
            set_loss = set_loss.cuda()
        loss = mseloss + set_loss

        return loss

def train(input_variable, target_variable, encoder, decoder, codes_inverse_freq, encoder_optimizer, decoder_optimizer,
          criterion, output_size, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = len(input_variable)
    target_length = len(target_variable)

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    if use_cuda:
        encoder_outputs = encoder_outputs.cuda()

    history_record = np.zeros(output_size)
    for ei in range(input_length - 1):
        if ei == 0: #because first basket in input variable is [-1]
            continue
        for ele in input_variable[ei]:
            history_record[ele] += 1.0 / (input_length - 2)

    for ei in range(input_length - 1):
        if ei == 0:
            continue
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei - 1] = encoder_output[0][0]

    last_input = input_variable[input_length - 2]
    decoder_hidden = encoder_hidden
    last_hidden = encoder_hidden
    decoder_input = last_input

    decoder_output, decoder_hidden, decoder_attention = decoder(
        decoder_input, decoder_hidden, encoder_outputs, history_record, last_hidden)

    #create target tensor.
    vectorized_target = np.zeros(output_size)
    for idx in target_variable[1]:
        vectorized_target[idx] = 1
    target = Variable(torch.FloatTensor(vectorized_target).reshape(1, -1))

    if use_cuda:
        target = target.cuda()
    weights = Variable(torch.FloatTensor(codes_inverse_freq).reshape(1, -1))
    if use_cuda:
        weights = weights.cuda()

    loss = criterion(decoder_output, target, weights)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(data_history, data_future, output_size, encoder, decoder, model_name, training_key_set, val_keyset, codes_inverse_freq, next_k_step,
               n_iters, top_k):
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    # elem_wise_connection.initWeight()

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-11,
                                         weight_decay=0)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-11,
                                         weight_decay=0)


    total_iter = 0
    criterion = custom_MultiLabelLoss_torch()
    best_recall = 0.0
    # train n_iters epoch
    for j in range(n_iters):
        # get a suffle list
        key_idx = np.random.permutation(len(training_key_set))
        training_keys = []
        for idx in key_idx:
            training_keys.append(training_key_set[idx])

        for iter in tqdm(range(0, len(training_key_set))):
            # get training data and label.
            input_variable = data_history[training_keys[iter]]
            target_variable = data_future[training_keys[iter]]

            loss = train(input_variable, target_variable, encoder,
                         decoder, codes_inverse_freq, encoder_optimizer, decoder_optimizer, criterion, output_size)

            print_loss_total += loss
            total_iter += 1

        # print loss and save model
        print_loss_avg = print_loss_total / len(training_key_set)
        print_loss_total = 0
        print('%s (%d %d%%) %.6f' % (timeSince(start, total_iter / (n_iters * len(training_key_set))), total_iter,
                                     total_iter / (n_iters * len(training_key_set)) * 100, print_loss_avg))
        sys.stdout.flush()

        recall, ndcg, hr = evaluate(data_history, data_future, encoder, decoder, output_size, val_keyset, next_k_step,
                 top_k)
        if recall>best_recall:
            best_recall=recall
            # print(pred_dict[user])
            filepath = './models/encoder_' + (model_name) + '_model_best'
            torch.save(encoder, filepath)
            filepath = './models/decoder_' + (model_name) + '_model_best'
            torch.save(decoder, filepath)
            print('Recall:', recall)
        print('Finish epoch: ' + str(j))
        print('Model is saved.')
######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.

cosine_sim = []
pair_cosine_sim = []

def decoding_next_k_step(encoder, decoder, input_variable, target_variable, output_size, k, activate_codes_num):
    # k is the next k step.
    encoder_hidden = encoder.initHidden()

    input_length = len(input_variable)
    encoder_outputs = Variable(torch.zeros(MAX_LENGTH, encoder.hidden_size))
    if use_cuda:
        encoder_outputs = encoder_outputs.cuda()

    # history frequency information
    history_record = np.zeros(output_size)
    count = 0.0
    for ei in range(input_length - 1):
        if ei == 0:
            continue
        for ele in input_variable[ei]:
            history_record[ele] += 1
        count += 1.0
    history_record = history_record / count

    # basket item iterator
    for ei in range(input_length - 1):
        if ei == 0:
            continue
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei - 1] = encoder_output[0][0]

        for ii in range(k):
            vectorized_target = np.zeros(output_size)
            for idx in target_variable[ii + 1]:
                vectorized_target[idx] = 1

            vectorized_input = np.zeros(output_size)
            for idx in input_variable[ei]:
                vectorized_input[idx] = 1

    decoder_input = input_variable[input_length - 2]
    decoder_hidden = encoder_hidden
    last_hidden = decoder_hidden
    topk = 400
    decoded_vectors = []
    prob_vectors = []
    # k is the number of steps need to predicted, for next basket is 1
    for di in range(k):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs, history_record, last_hidden)
        # topv is top values, topi is top indicies.
        topv, topi = decoder_output.data.topk(topk)

        # construct target vector
        vectorized_target = np.zeros(output_size)
        for idx in target_variable[di + 1]:# iter the target basket.
            vectorized_target[idx] = 1

        count = 0
        if activate_codes_num > 0:
            pick_num = activate_codes_num
        else:
            pick_num = np.sum(vectorized_target)

        tmp = []
        for ele in range(len(topi[0])):
            if count >= pick_num:
                break
            tmp.append(topi[0][ele])
            count += 1
        decoded_vectors.append(tmp)
        decoder_input = tmp

        tmp = []
        for i in range(topk):
            tmp.append(topi[0][i])
        prob_vectors.append(tmp)

    return decoded_vectors, prob_vectors


def get_precision_recall_Fscore(groundtruth, pred):
    a = groundtruth
    b = pred
    correct = 0
    truth = 0
    positive = 0

    for idx in range(len(a)):
        if a[idx] == 1:
            truth += 1
            if b[idx] == 1:
                correct += 1
        if b[idx] == 1:
            positive += 1

    flag = 0
    if 0 == positive:
        precision = 0
        flag = 1
        # print('postivie is 0')
    else:
        precision = correct / positive
    if 0 == truth:
        recall = 0
        flag = 1
        # print('recall is 0')
    else:
        recall = correct / truth

    if flag == 0 and precision + recall > 0:
        F = 2 * precision * recall / (precision + recall)
    else:
        F = 0
    return precision, recall, F, correct


def get_F_score(prediction, test_Y):
    jaccard_similarity = []
    prec = []
    rec = []

    count = 0
    for idx in range(len(test_Y)):
        pred = prediction[idx]
        T = 0
        P = 0
        correct = 0
        for id in range(len(pred)):
            if test_Y[idx][id] == 1:
                T = T + 1
                if pred[id] == 1:
                    correct = correct + 1
            if pred[id] == 1:
                P = P + 1

        if P == 0 or T == 0:
            continue
        precision = correct / P
        recall = correct / T
        prec.append(precision)
        rec.append(recall)
        if correct == 0:
            jaccard_similarity.append(0)
        else:
            jaccard_similarity.append(2 * precision * recall / (precision + recall))
        count = count + 1

    print(
        'average precision: ' + str(np.mean(prec)))
    print('average recall : ' + str(
        np.mean(rec)))
    print('average F score: ' + str(
        np.mean(jaccard_similarity)))


def get_DCG(groundtruth, pred_rank_list, k):
    count = 0
    dcg = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if groundtruth[pred] == 1:
            dcg += (1) / math.log2(count + 1 + 1)
        count += 1

    return dcg


def get_NDCG(groundtruth, pred_rank_list, k):
    count = 0
    dcg = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if groundtruth[pred] == 1:
            dcg += (1) / math.log2(count + 1 + 1)
        count += 1
    idcg = 0
    num_real_item = np.sum(groundtruth)
    num_item = int(min(num_real_item, k))
    for i in range(num_item):
        idcg += (1) / math.log2(i + 1 + 1)
    ndcg = dcg / idcg
    return ndcg


def get_HT(groundtruth, pred_rank_list, k):
    count = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if groundtruth[pred] == 1:
            return 1
        count += 1

    return 0


def evaluate(history_data, future_data, encoder, decoder, output_size, test_key_set, next_k_step, activate_codes_num):
    #activate_codes_num: pick top x as the basket.
    prec = []
    rec = []
    F = []
    prec1 = []
    rec1 = []
    F1 = []
    prec2 = []
    rec2 = []
    F2 = []
    prec3 = []
    rec3 = []
    F3 = []

    NDCG = []
    n_hit = 0
    count = 0

    for iter in range(len(test_key_set)):
        # training_pair = training_pairs[iter - 1]
        # input_variable = training_pair[0]
        # target_variable = training_pair[1]
        input_variable = history_data[test_key_set[iter]]
        target_variable = future_data[test_key_set[iter]]

        if len(target_variable) < 2 + next_k_step:
            continue
        count += 1
        output_vectors, prob_vectors = decoding_next_k_step(encoder, decoder, input_variable, target_variable,
                                                            output_size, next_k_step, activate_codes_num)

        hit = 0
        for idx in range(len(output_vectors)):
            # for idx in [2]:
            vectorized_target = np.zeros(output_size)
            for ii in target_variable[1 + idx]: #target_variable[[-1], [item, item], .., [-1]]
                vectorized_target[ii] = 1

            vectorized_output = np.zeros(output_size)
            for ii in output_vectors[idx]:
                vectorized_output[ii] = 1

            precision, recall, Fscore, correct = get_precision_recall_Fscore(vectorized_target, vectorized_output)
            prec.append(precision)
            rec.append(recall)
            F.append(Fscore)
            if idx == 0:
                prec1.append(precision)
                rec1.append(recall)
                F1.append(Fscore)
            elif idx == 1:
                prec2.append(precision)
                rec2.append(recall)
                F2.append(Fscore)
            elif idx == 2:
                prec3.append(precision)
                rec3.append(recall)
                F3.append(Fscore)
            # length[idx] += np.sum(target_variable[1 + idx])
            # prob_vectors is the probability
            target_topi = prob_vectors[idx]
            hit += get_HT(vectorized_target, target_topi, activate_codes_num)
            ndcg = get_NDCG(vectorized_target, target_topi, activate_codes_num)
            NDCG.append(ndcg)
        if hit == next_k_step:
            n_hit += 1

    return np.mean(rec), np.mean(NDCG), n_hit / len(test_key_set)


def get_codes_frequency_no_vector(history_data, num_dim, key_set):
    result_vector = np.zeros(num_dim)
    #pid is users id
    for pid in key_set:
        for idx in history_data[pid]:
            if idx == [-1]:
                continue
            result_vector[idx] += 1
    return result_vector


def main(argv):

    directory = './amodels/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    dataset = argv[1]
    ind = argv[2]
    history_file = '../../jsondata/'+dataset+'_history.json'
    future_file = '../../jsondata/'+dataset+'_future.json'
    keyset_file = '../../keyset/'+dataset+'_keyset_'+str(ind)+'.json'
    model_version = dataset+str(ind)
    topk = int(argv[3])
    training = int(argv[4])

    next_k_step = 1
    with open(history_file, 'r') as f:
        history_data = json.load(f)
    with open(future_file, 'r') as f:
        future_data = json.load(f)
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)

    input_size = keyset['item_num']
    training_key_set = keyset['train']
    val_key_set = keyset['val']
    test_key_set = keyset['test']

    # weights is inverse personal top frequency. normalized by max freq.
    weights = np.zeros(input_size)
    codes_freq = get_codes_frequency_no_vector(history_data, input_size, future_data.keys())
    max_freq = max(codes_freq)
    for idx in range(len(codes_freq)):
        if codes_freq[idx] > 0:
            weights[idx] = max_freq / codes_freq[idx]
        else:
            weights[idx] = 0

    # Sets2sets model
    encoder = EncoderRNN_new(input_size, hidden_size, num_layers)
    attn_decoder = AttnDecoderRNN_new(hidden_size, input_size, num_layers, dropout_p=0.1)
    if use_cuda:
        encoder = encoder.cuda()
        attn_decoder = attn_decoder.cuda()

    # train mode or test mode
    if training == 1:
        trainIters(history_data, future_data, input_size, encoder, attn_decoder, model_version, training_key_set, val_key_set, weights,
                   next_k_step, num_iter, topk)

    else:
        for i in [10, 20]: #top k
            valid_recall = []
            valid_ndcg = []
            valid_hr = []
            recall_list = []
            ndcg_list = []
            hr_list = []
            print('k = ' + str(i))
            for model_epoch in range(num_iter):
                print('Epoch: ', model_epoch)
                encoder_pathes = './models/encoder' + str(model_version) + '_model_epoch' + str(model_epoch)
                decoder_pathes = './models/decoder' + str(model_version) + '_model_epoch' + str(model_epoch)

                encoder_instance = torch.load(encoder_pathes, map_location=torch.device('cpu'))
                decoder_instance = torch.load(decoder_pathes, map_location=torch.device('cpu'))

                recall, ndcg, hr = evaluate(history_data, future_data, encoder_instance, decoder_instance, input_size,
                                            val_key_set, next_k_step, i)
                valid_recall.append(recall)
                valid_ndcg.append(ndcg)
                valid_hr.append(hr)
                recall, ndcg, hr = evaluate(history_data, future_data, encoder_instance, decoder_instance, input_size,
                                            test_key_set, next_k_step, i)
                recall_list.append(recall)
                ndcg_list.append(ndcg)
                hr_list.append(hr)
            valid_recall = np.asarray(valid_recall)
            valid_ndcg = np.asarray(valid_ndcg)
            valid_hr = np.asarray(valid_hr)
            idx1 = valid_recall.argsort()[::-1][0]
            idx2 = valid_ndcg.argsort()[::-1][0]
            idx3 = valid_hr.argsort()[::-1][0]
            print('max valid recall results:')
            print('Epoch: ', idx1)
            print('recall: ', recall_list[idx1])
            print('ndcg: ', ndcg_list[idx1])
            print('phr: ', hr_list[idx1])
            sys.stdout.flush()

            print('max valid ndcg results:')
            print('Epoch: ', idx2)
            print('recall: ', recall_list[idx2])
            print('ndcg: ', ndcg_list[idx2])
            print('phr: ', hr_list[idx2])
            sys.stdout.flush()

            print('max valid phr results:')
            print('Epoch: ', idx3)
            print('recall: ', recall_list[idx3])
            print('ndcg: ', ndcg_list[idx3])
            print('phr: ', hr_list[idx3])
            sys.stdout.flush()


if __name__ == '__main__':
    main(sys.argv)
