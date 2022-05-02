import math

import numpy as np
import random
import sys
import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from pytorch_metric import *

class Wloss(nn.modules.loss._Loss):

    def __init__(self, p, n):
        super(Wloss, self).__init__()
        self.p = p
        self.n = n
        if p > n:
            self.mode = 'positive'
        else:
            self.mode = 'negative'

    def forward(self, pred, tgt, cand):
        # pred is the score vector, tgt is the label list.
        loss = 0.0
        if self.mode == 'positive':
            for ind in range(pred.size(0)):
                if ind in tgt:
                    loss += -torch.log(pred[ind])*self.p
                else:
                    loss += -torch.log(1-pred[ind])*self.n

        if self.mode == 'negative':
            for ind in range(pred.size(0)):
                if ind in tgt:
                    loss += -torch.log(pred[ind])*self.p
                else:
                    if ind in cand:
                        loss += -torch.log(1-pred[ind])*self.n
                    else:
                        loss += -torch.log(1-pred[ind])
        return loss/pred.size(0)

class NBRNet(nn.Module):

    def __init__(self, config, dataset):
        super(NBRNet, self).__init__()

        # device setting
        self.device = config['device']

        # dataset features
        self.n_items = dataset['item_num']

        # model parameters
        self.embedding_size = config['embedding_size']
        self.embedding_type = config['embedding_type']
        self.hidden_size = config['hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.max_len = config['max_len'] # basket len
        self.loss_mode = config['loss_mode']
        self.loss_uplift = config['loss_uplift']
        # define layers
        # self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, self.max_len)
        self.basket_embedding = Basket_Embedding(self.device, self.embedding_size, self.n_items, self.max_len, self.embedding_type)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)
        self.attention = config['attention']
        self.decoder = Decoder(
            self.device,
            hidden_size=self.hidden_size,
            seq_len=self.max_len,
            num_item=self.n_items,
            dropout_prob=self.dropout_prob,
            attention=self.attention
        )


        self.loss_fct = nn.BCELoss()
        self.p_loss_fct = Wloss(self.loss_uplift, 1)
        self.n_loss_fct = Wloss(1, self.loss_uplift)
        self.meta_loss_fct = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, basket_seq, candidates_basket):
        basket_seq_len = []
        for b in basket_seq:
            basket_seq_len.append(len(b))
        basket_seq_len = torch.as_tensor(basket_seq_len).to(self.device)

        batch_basket_seq_embed = self.basket_embedding(basket_seq)

        all_memory, _ = self.gru(batch_basket_seq_embed)
        last_memory = self.gather_indexes(all_memory, basket_seq_len-1)

        timeline_mask = get_timeline_mask(batch_basket_seq_embed, self.device, self.embedding_size)
        # Need to get the candidates for decoder, Here we condiser the candidates could be pre-calculated.
        # including repeat_candidates, explore(exculde repeated items), explore(item, user, popular)
        pred = self.decoder.forward(all_memory, last_memory, candidates_basket, timeline_mask)
        return pred


    def predict(self):
        pass

    def get_batch_loss(self, pred, tgt, cand, tag, device):
        # need to handle the case that
        batch_size = pred.size(0)
        tmp_tgt = get_label_tensor(tgt, device, self.n_items)
        loss = 0.0
        if self.loss_mode == 0:
            for ind in range(batch_size):
                pred_ind = torch.clamp(pred[ind], 0.001, 0.999)
                loss += self.loss_fct(pred_ind.unsqueeze(0), tmp_tgt[ind].unsqueeze(0))
        if self.loss_mode == 1:
            if tag == 'negative':
                for ind in range(batch_size):
                    user_pred_ind = torch.clamp(pred[ind], 0.001, 0.999)
                    user_tgt = torch.tensor(tgt[ind])
                    user_cand = torch.tensor(cand[ind])
                    loss += self.n_loss_fct(user_pred_ind, user_tgt, user_cand)
            if tag == 'positive':
                for ind in range(batch_size):
                    user_pred_ind = torch.clamp(pred[ind], 0.001, 0.999)
                    user_tgt = torch.tensor(tgt[ind])
                    user_cand = torch.tensor(cand[ind])
                    loss += self.p_loss_fct(user_pred_ind, user_tgt, user_cand)
        return loss/batch_size # compute average


    def global_loss(self, basket_seq, tgt_basket, cand_basket):
        prediction = self.forward(basket_seq, cand_basket)
        cand = [l1+l2 for l1, l2 in zip(cand_basket['repeat'], cand_basket['explore'])]
        loss = self.get_batch_loss(prediction,
                                   tgt_basket,
                                   cand,
                                   'positive',
                                   self.device) #the multilabel loss here
        return loss

    def calculate_loss(self, basket_seq, tgt_basket, cand_basket):
        global_loss = self.global_loss(basket_seq, tgt_basket, cand_basket)
        return global_loss


    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

# Provide basket embedding solution: max, mean, sum
class Basket_Embedding(nn.Module):

    def __init__(self, device, hidden_size, item_num, max_len, type): # hidden_size is the embedding_size
        super(Basket_Embedding, self).__init__()
        self.hidden_size = hidden_size
        self.n_items = item_num
        self.max_len = max_len
        self.type = type
        self.device = device
        self.item_embedding = nn.Embedding(item_num, hidden_size) # padding_idx=0, not sure???

    def forward(self, batch_basket):
        # need to padding here
        batch_embed_seq = [] # batch * seq_len * hidden size
        for basket_seq in batch_basket:
            embed_baskets = []
            for basket in basket_seq:
                basket = torch.LongTensor(basket).resize_(1, len(basket))
                basket = Variable(basket).to(self.device)
                basket = self.item_embedding(basket).squeeze(0)
                # embed_b = basket_pool(basket, 1, self.type)
                if self.type == 'mean':
                    embed_baskets.append(torch.mean(basket, 0))
                if self.type == 'max':
                    embed_baskets.append(torch.max(basket, 0)[0])
                if self.type == 'sum':
                    embed_baskets.append(torch.sum(basket, 0))
            # padding the seq
            pad_num = self.max_len -len(embed_baskets)
            for ind in range(pad_num):
                embed_baskets.append(torch.zeros(self.hidden_size).to(self.device))
            embed_seq = torch.stack(embed_baskets, 0)
            embed_seq = torch.as_tensor(embed_seq)
            batch_embed_seq.append(embed_seq)
        batch_embed_output = torch.stack(batch_embed_seq, 0).to(self.device)
        return batch_embed_output


class Decoder(nn.Module):
    def __init__(self, device, hidden_size, seq_len, num_item, dropout_prob, attention):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.device = device
        self.seq_len = seq_len
        self.n_items = num_item
        self.attention = attention

        if self.attention == 'attention':
            self.W_repeat = nn.Linear(hidden_size, hidden_size, bias=False)
            self.U_repeat = nn.Linear(hidden_size, hidden_size, bias=False)
            self.tanh = nn.Tanh()
            self.V_repeat = nn.Linear(hidden_size, 1)
            self.Repeat = nn.Linear(hidden_size*2, num_item)
        else:
            self.Repeat = nn.Linear(hidden_size, num_item)
        # self.softmax = nn.softmax
        # self.sigmoid = nn.Sigmoid()


    def forward(self, all_memory, last_memory, item_seq, mask=None):
        '''item_seq is the appared items or candidate items'''
        if self.attention == 'attention':
            all_memory_values, last_memory_values = all_memory, last_memory
            all_memory = self.dropout(self.U_repeat(all_memory))
            last_memory = self.dropout(self.W_repeat(last_memory))
            last_memory = last_memory.unsqueeze(1)
            last_memory = last_memory.repeat(1, self.seq_len, 1)

            output_er = self.tanh(all_memory+last_memory)
            output_er = self.V_repeat(output_er).squeeze(-1)

            if mask is not None:
                output_er.masked_fill_(mask, -1e9)

            output_er = output_er.unsqueeze(-1)

            alpha_r = nn.Softmax(dim=1)(output_er)
            alpha_r = alpha_r.repeat(1, 1, self.hidden_size)
            output_r = (all_memory_values*alpha_r).sum(dim=1)
            output_r = torch.cat([output_r, last_memory_values], dim=1)
            output_r = self.dropout(self.Repeat(output_r))

            # repeat_mask = get_candidate_mask(item_seq, self.device, self.n_items)
            # output_r = output_r.masked_fill(repeat_mask.bool(), float('-inf'))
            decoder = torch.sigmoid(output_r)
        else:
            decoder = torch.sigmoid(self.dropout(self.Repeat(last_memory)))

        return decoder


def get_candidate_mask(candidates, device, max_index=None):
    '''Candidates is the output of basic models or repeat or popular'''
    batch_size = len(candidates)
    if torch.cuda.is_available():
        candidates_mask = torch.FloatTensor(batch_size, max_index).fill_(1.0).to(device)
    else:
        candidates_mask = torch.ones(batch_size, max_index)
    for ind in range(batch_size):
        candidates_mask[ind].scatter_(0, torch.as_tensor(candidates[ind]).to(device), 0)
    candidates_mask.requires_grad = False
    return candidates_mask.bool()

def get_timeline_mask(batch_basket_emb, device, emb_size):
    batch_mask = []
    for basket_seq in batch_basket_emb:
        mask = []
        for basket_emb in basket_seq:
            if torch.equal(basket_emb, torch.zeros(emb_size).to(device)):
                mask.append(1)
            else:
                mask.append(0)
        batch_mask.append(torch.as_tensor(mask).bool())
    batch_mask = torch.stack(batch_mask, 0).to(device)
    return batch_mask.bool()

def get_label_tensor(labels, device, max_index=None):
    '''Candidates is the output of basic models or repeat or popular
    labels is list[]'''
    batch_size = len(labels)
    if torch.cuda.is_available():
        label_tensor = torch.FloatTensor(batch_size, max_index).fill_(0.0).to(device)
    else:
        label_tensor = torch.zeros(batch_size, max_index)
    for ind in range(batch_size):
        if len(labels[ind])!=0:
            label_tensor[ind].scatter_(0, torch.as_tensor(labels[ind]).to(device), 1)
    label_tensor.requires_grad = False # because this is not trainable
    return label_tensor


def get_sub_label_set(labels, candidates):
    batch_size = len(labels)
    batch_label_set = []
    # print('Labels', labels[0])
    # print("Candiates:", candidates[0])
    for ind in range(batch_size):
        # sub_labels = list(set(labels[ind]&candidates[ind]))
        sub_labels = [item for item in labels[ind] if item in candidates[ind]]
        batch_label_set.append(sub_labels)
    # batch_sub_labels = torch.stack(batch_label_set, dim=0).to(device)
    # print('cand:', batch_label_set[0])
    return batch_label_set