
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class NBModel(nn.Module):  #
    def __init__(self, config, device):
        super(NBModel, self).__init__()
        self.device = device
        self.sd1 = config.sd1
        self.sd2 = config.sd2
        self.sd3 = config.sd3
        self.sd4 = config.sd4
        self.sd5 = config.sd5

        self.neg_margin = config.neg_margin
        self.pos_margin = config.pos_margin

        self.num_users = config.num_users

        self.num_product = config.num_product

        self.D = Discriminator(config, self.device)
        self.G0 = Generator1(config, self.device)
        self.G2 = Generator2(config, self.device)

        self.G1_flag = 0
        self.mse = nn.MSELoss()

    def init_weight(self):
        torch.nn.init.xavier_normal_(self.embed.weight)

    # profile
    def forward(self, T, uid, input_seq_tensor, tar_b_tensor, weight, history, neg_set_tensor=None, train=True,
                G1flag=0, pretrain=0, sd2=1):
        '''

        :param T:  is  Gumbel softmax's temperature
        :param uid:  is  userid
        :param input_seq_tensor:   K * B
        :param tar_b_tensor:  K
        :param weight:  itemnum
        :param history:  K * itemnum
        :param neg_set_tensor: K * neg_ratio
        :param train: whether train
        :return: classify prob metric K * itemnum
        '''

        self.G1_flag = G1flag
        self.pretrain = pretrain

        mask = torch.ones_like(input_seq_tensor,dtype = torch.float).to(self.device)
        mask[input_seq_tensor == -1] = 0
        tar_expand = tar_b_tensor.view(-1,1).expand_as(input_seq_tensor)
        mask0 = torch.zeros_like(input_seq_tensor,dtype = torch.float).to(self.device)
        mask0[tar_expand == input_seq_tensor] = 1

        input_embeddings = self.G2.embed1(input_seq_tensor + 1)
        target_embedding = self.G2.embed1(tar_b_tensor + 1)

        test = 1
        if train == True:
            test = 0
        # print(self.embed.weight.data)
        if ((self.G1_flag == 0) & (pretrain == 0)):  #
            #
            self.filter_basket = torch.ones_like(input_seq_tensor,dtype = torch.float).to(self.device)  # K * B
            real_generate_embedding1 = self.G2(self.filter_basket, input_seq_tensor,uid)  # K*H
            fake_discr = self.D(real_generate_embedding1, history, tar_b_tensor)  # K*n_items

            if train == True:
                all_sum = mask.sum()

                loss, (p1, p2, p3, p4) = self.loss2_G1flag0(fake_discr, tar_b_tensor)

                return loss, fake_discr, (p1, p2, p3, p4), (all_sum, all_sum, all_sum, all_sum/all_sum,all_sum/all_sum)

            fake_discr = torch.softmax(fake_discr, dim=-1)
            return mask.sum() / mask.sum(),mask0.sum() / mask0.sum(), fake_discr
        else:
            self.filter_basket, test_basket = self.G0(input_seq_tensor, T, tar_b_tensor, self.G1_flag,test,input_embeddings,target_embedding)  # K * B
            real_generate_embedding1 = self.G2(self.filter_basket[:, :, 0], input_seq_tensor,uid)
            fake_discr = self.D(real_generate_embedding1, history, tar_b_tensor, input_seq_tensor)  # K*n_items


            ################################################
            select_repeats = mask0 * ((self.filter_basket[:,:,0] > 1 / 2 ).float())#torch.tensor((self.filter_basket[:,:,0] > 1 / 2 ).int(),dtype = torch.long).to(self.device)

            repeat_ratio = (select_repeats.sum(1)/(mask0.sum(1)+1e-24)).sum()/(((mask0.sum(1)>0).float()+1e-24).sum())

            if train == True:

                rest = mask * self.filter_basket[:, :, 0]#.detach()
                rest_int = (rest > 1 / 2).int()
                real_rest_sum = rest.sum()
                real_rest_sum_int = rest_int.sum()
                all_sum = mask.sum()
                ratio = (rest * ((rest > 1 / 2).float())).sum() / max(1, ((rest > 1 / 2).float()).sum())

                # self.rest_basket = torch.ones_like(self.filter_basket,dtype = torch.float).to(self.device) - self.filter_basket
                rest_generate_embedding1 = self.G2(self.filter_basket[:, :, 1], input_seq_tensor,uid)
                rest_discr = self.D(rest_generate_embedding1, history, tar_b_tensor, input_seq_tensor)

                filter_pos = mask * self.filter_basket[:, :, 0]
                filter_neg = mask * self.filter_basket[:, :, 1]

                self.whole_basket = torch.ones_like(input_seq_tensor,dtype = torch.float).to(self.device)  # K * B
                whole_generate_embedding1 = self.G2(self.whole_basket, input_seq_tensor,uid)  # K*H
                whole_discr = self.D(whole_generate_embedding1, history, tar_b_tensor)  # K*n_items

                loss, (p1, p2, p3, p4) = self.loss2_G1flag1(fake_discr, tar_b_tensor, rest_discr,
                                                                 whole_discr, filter_pos, filter_neg,
                                                                 mask)
                return loss, fake_discr, (p1, p2, p3, p4), (real_rest_sum, real_rest_sum_int, all_sum, ratio, repeat_ratio)

            select_repeats = mask0 * ((test_basket > 1 / 2).float()) #torch.tensor((test_basket > 1 / 2).int(),dtype = torch.long).to(self.device)
            test_repeat_ratio = (select_repeats.sum(1) / (mask0.sum(1) + 1e-24)).sum() / (((mask0.sum(1) > 0).float()).sum())
            # test_repeat_ratio = torch.mean(select_repeats.sum(1) / (mask0.sum(1) + 1e-24))

            rest_test = mask * test_basket.detach()
            rest_sum = rest_test.sum()
            all_sum = mask.sum()
            test_rest_ratio = rest_sum / all_sum
            test_generate_embedding1 = self.G2(test_basket, input_seq_tensor,uid)
            test_discr = self.D(test_generate_embedding1, history, tar_b_tensor, input_seq_tensor)  # K*n_items
            test_discr = torch.softmax(test_discr, -1)
            return test_rest_ratio, test_repeat_ratio, test_discr

    def loss2_G1flag0(self, fake_discr, target_labels):
        '''
        :param fake_discr:   K * itemnum
        :param target_labels:  K
        :param neg_labels:  K * neg_ratio = K * nK
        :return:
        '''
        fake_discr = torch.softmax(fake_discr, dim=-1)

        item_num = fake_discr.size(0)  # K
        index = torch.tensor(np.linspace(0, item_num, num=item_num, endpoint=False), dtype=torch.long)  # K
        pfake = fake_discr[index, target_labels]  # K

        loss_1 = - torch.mean(torch.log((pfake) + 1e-24))

        loss = self.sd1 * loss_1
        return loss, (loss_1, loss_1, loss_1, loss_1)

    def loss2_G1flag1(self, fake_discr, target_labels, rest_discri,whole_discr,filter_pos,filter_neg,mask):
        '''
        :param fake_discr:   K * itemnum
        :param filter_pos:   K * B
        :param target_labels:  K
        :param neg_labels:  K * neg_ratio = K * nK
        :param rest_discri:  K * itemnum
        :param neg_discri:  (Kxjudge_ratio) * itemnum
        :return:
        '''
        fake_discr = torch.softmax(fake_discr,dim = -1)
        rest_discri = torch.softmax(rest_discri,dim = -1)
        whole_discr = torch.softmax(whole_discr, dim=-1)


        item_num = fake_discr.size(0)  # K
        index = torch.tensor(np.linspace(0, item_num, num=item_num, endpoint=False), dtype=torch.long)  # K
        pfake = fake_discr[index, target_labels]  # K
        prest = rest_discri[index, target_labels]  # K
        pwhole = whole_discr[index, target_labels]  # K

        pos_restratio = torch.mean(filter_pos.sum(1).view(1,-1)/(mask.sum(1).view(1,-1)),dim=-1).view(1,-1) # 1*1
        pos_margin = self.pos_margin * torch.ones_like(pos_restratio).to(self.device)  #
        zeros = torch.zeros_like(pos_restratio).to(self.device)
        pos_restratio = torch.cat((pos_margin-pos_restratio,zeros),dim=0) #2*1

        neg_restratio = torch.mean(filter_neg.sum(1).view(1,-1)/(mask.sum(1).view(1,-1)),dim=-1).view(1,-1) # 1*1
        neg_margin = self.neg_margin * torch.ones_like(neg_restratio).to(self.device)
        neg_restratio = torch.cat((neg_restratio - neg_margin, zeros), dim=0)

        loss_0 = torch.max(pos_restratio,dim = 0)[0] + torch.max(neg_restratio,dim = 0)[0]
        loss_1 = - torch.mean(torch.nn.LogSigmoid()(pfake - pwhole))
        loss_2 = - torch.mean(torch.nn.LogSigmoid()(pwhole - prest))
        loss_3 = - torch.mean(torch.log(pfake + 1e-24))
        loss0 = loss_0 + loss_1 + loss_2 + loss_3
        if (self.G1_flag == 1):
            return loss0, (loss_1, loss_2, loss_3, loss_3)
        else:
            loss_4 = - torch.mean(torch.log((pwhole + 1e-24) ))
            loss0 = loss_0 + loss_1 + loss_2 + loss_3 + loss_4
        return loss0, (loss_1, loss_2, loss_3, loss_4)

class Generator1(nn.Module):
    def __init__(self, config, device, dropout_p=0.2):
        super(Generator1, self).__init__()
        self.device = device
        self.dropout_p = config.dropout
        self.input_size = config.num_product
        self.hidden_size = config.embedding_dim
        self.max_basket_size = config.max_basket_size

        self.same_embedding = config.same_embedding

        self.temp_learn = config.temp_learn
        self.temp = nn.Parameter(torch.ones(1)* config.temp)
        self.temp_init = config.temp

        self.embed = nn.Embedding(config.num_product + 1, self.hidden_size, padding_idx=0)
        # self.W = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.judge_model = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Dropout(self.dropout_p),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_size, 2)
        )
        self.judge_model1 = nn.Linear(self.hidden_size * 2, 2)

    # profile
    def init_weight(self):
        for name, parms in self.named_parameters():  # TODO
            parms.data.normal_(0, 0.1)
        torch.nn.init.xavier_normal_(self.embed.weight.data)  # good
        self.temp = nn.Parameter(torch.ones(1).to(self.device)* self.temp_init)  # * config.temp # TODO  这个必须弄，不然templearn=1时它一开始rest_ratio = 0.99

    def forward(self, input_seq_tensor, T, target_tensor, G1_flag=1,test=0,input_embeddings = None,target_embedding = None):
        '''
        :param input_seq_tensor:  K * B
        :param T:
        :param target_tensor:
        :param G1_flag:
        :param test:
        :param input_embeddings:
        :param target_embedding:
        :return:
        '''
        def hook_fn(grad):
            print(grad)


        # target_embedding = self.embed(target_tensor + 1)
        # input_embeddings = self.embed(input_seq_tensor + 1)
        # target_embedding = self.W(target_embedding)
        # input_embeddings = self.W(input_embeddings)
        if self.same_embedding == 0:
            target_embedding = self.embed(target_tensor + 1)
            input_embeddings = self.embed(input_seq_tensor + 1)

        in_tar = torch.cat(
            (input_embeddings, target_embedding.view(target_embedding.size(0), 1, -1).expand_as(input_embeddings)),
            dim=2)  # K*B*2H
        in_tar = self.dropout(in_tar)
        resnet_o_prob = self.judge_model1(in_tar)
        o_prob = self.judge_model(in_tar)  # K*B*2

        o_prob = (o_prob + resnet_o_prob)  #
        # att_prob = torch.sigmoid(o_prob)
        o_prob = torch.softmax(o_prob, dim=-1)


        if self.temp_learn == 1:
            if self.temp > 0:
                prob_hard, prob_soft = self.gumbel_softmax(torch.log(o_prob + 1e-24), self.temp, hard=True,input_seq_tensor = input_seq_tensor)
            else:
                prob_hard, prob_soft = self.gumbel_softmax(torch.log(o_prob + 1e-24), 0.3, hard=True,input_seq_tensor = input_seq_tensor)
        else:
            prob_hard, prob_soft = self.gumbel_softmax(torch.log(o_prob + 1e-24), T, hard=True,input_seq_tensor = input_seq_tensor)

        prob_soft_new = prob_hard*prob_soft

        # return prob_soft_new, prob_soft_new[:,:,0]

        if test == 0 :#and G1_flag != 0:
            return prob_soft_new, None
        else:
            if self.temp_learn == 1:
                if self.temp > 0:
                    o_prob_hard, o_prob_soft = self.gumbel_test(torch.log(o_prob + 1e-24), self.temp)
                else:
                    o_prob_hard, o_prob_soft = self.gumbel_test(torch.log(o_prob + 1e-24), 0.3)
            else:
                o_prob_hard, o_prob_soft = self.gumbel_test(torch.log(o_prob + 1e-24), T)

            test_prob_hard = o_prob_hard * o_prob_soft
            test_prob_hard = test_prob_hard.detach()

        return test_prob_hard, test_prob_hard[:, :, 0]

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).to(self.device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature,input_seq_tensor = None):
        '''
        :param logits: # K*B*2
        :param temperature:
        :param input_seq_tensor:  K*B
        :return:
        '''
        sample = self.sample_gumbel([int(self.input_size+1),2],eps= 1e-20) # n_items+1  *  2
        x_index = input_seq_tensor.clone()+1 #K*B
        x_index = x_index.unsqueeze(2).repeat(1,1,2) # K*B*2
        # print(x_index.size())
        y_index = torch.zeros_like(input_seq_tensor,dtype = torch.long).to(self.device).unsqueeze(2) #K*B*1
        y_index1 = torch.ones_like(input_seq_tensor, dtype=torch.long).to(self.device).unsqueeze(2)  # K*B*1
        y_index = torch.cat((y_index,y_index1),dim = 2) #K*B*2
        # print(y_index.size())
        sample_logits = sample[x_index.long(),y_index.long()]
        y = logits + sample_logits
        # y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False,input_seq_tensor = None):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature,input_seq_tensor)

        if not hard:
            return y  # .view(-1, latent_dim * categorical_dim)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        # y_hard = (y_hard - y).detach() + y
        return y_hard, y  # .view(-1, latent_dim * categorical_dim)

    def gumbel_test(self, logits, temperature):
        y = logits
        y = F.softmax(y / temperature, dim=-1)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return y_hard, y  # .view(-1, latent_dim * categorical_dim)

class Generator2(nn.Module):
    def __init__(self, config, device, dropout_p=0.2):
        super(Generator2, self).__init__()
        self.device = device
        self.dropout_p = config.dropout
        self.input_size = config.num_product
        self.num_users = config.num_users
        self.hidden_size = config.embedding_dim
        self.basket_pool_type = config.basket_pool_type

        self.dropout = nn.Dropout(self.dropout_p)
        self.max_basket_size = config.max_basket_size
        self.bidirectional = False
        self.batch_first = True
        self.num_layer = config.num_layer
        self.gru_hidden_size = self.hidden_size

        self.embed1 = nn.Embedding(config.num_product + 1, self.gru_hidden_size, padding_idx=0)

        self.gru1 = nn.GRU(self.gru_hidden_size,
                          self.gru_hidden_size,
                          num_layers=self.num_layer,
                          bidirectional=self.bidirectional,
                          batch_first=self.batch_first)


    # profile
    def forward(self, filter_basket_prob, input_seq_tensor,uid = None):
        '''
        :param filter_basket_prob:   K * B  [13] [2]  [0]  2
        :param input_embeddings:  K * B * H
        :param input_seq_tensor:  K * B  ,with padding -1
        :return: fake_target_embeddings  # K * hidden_size
        # basket_embedding  K*basket_num*H
        '''

        def hook_fn(grad):
            print(grad)

        mask = (input_seq_tensor != -1).detach().int()
        mask = torch.tensor(mask,dtype = torch.float).to(self.device)
        filter_basket_prob = filter_basket_prob * mask  # K * B
        # K*B*H

        input_embeddings1 = self.embed1(input_seq_tensor + 1)



        input_embeddings_f1 = torch.mul(input_embeddings1,
                                       filter_basket_prob.view(filter_basket_prob.size(0), -1, 1).expand(
                                           filter_basket_prob.size(0), -1, self.gru_hidden_size))

        input_embeddings_f1 = input_embeddings_f1.view(input_embeddings_f1.size(0), -1, self.max_basket_size,
                                                     self.gru_hidden_size)  # K*basket_num*max_basket_size*H

        filter_b = torch.max(filter_basket_prob.view(filter_basket_prob.size(0), -1, self.max_basket_size), dim=-1)[
            0]  # K*basket_num  ####

        if self.basket_pool_type == 'avg':
            filtered_tensor = filter_basket_prob.view(filter_basket_prob.size(0), -1, 1).expand(
                filter_basket_prob.size(0), -1, self.gru_hidden_size).view(input_embeddings_f1.size(0), -1,
                                                                       self.max_basket_size,
                                                                       self.gru_hidden_size)
            basket_embedding1 = (torch.sum(input_embeddings_f1, dim=2) / (
                    filtered_tensor.sum(dim=2) + 1e-10))  # K*basket_num*H
        else:
            mask_inf = filter_basket_prob.view(filter_basket_prob.size(0), -1, 1).expand(
                filter_basket_prob.size(0), -1, self.gru_hidden_size).int()
            mask_inf = (1 - mask_inf) * (-9999)
            mask_inf = mask_inf.view(mask_inf.size(0), -1, self.max_basket_size,
                                     self.gru_hidden_size)
            input_embeddings_f1 = input_embeddings_f1 + mask_inf
            basket_embedding1 = (torch.max(input_embeddings_f1, dim=2)[0])  # K*basket_num*H

        input_filter_b = (filter_b > 0).detach().int()  # K*basket_num ( value:0/1)
        sorted, indices = torch.sort(input_filter_b, descending=True)
        lengths = torch.sum(sorted, dim=-1).squeeze().view(1, -1).squeeze(0)
        length_mask = (lengths == 0).int()
        length_mask = torch.tensor(length_mask, dtype=torch.long).to(self.device)
        lengths = lengths + length_mask
        inputs1 = basket_embedding1.gather(dim=1,
                                           index=indices.unsqueeze(2).expand_as(
                                                   basket_embedding1))  # K*basket_num*H

        # sort data by lengths
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        sort_embed_input1 = inputs1.index_select(0, Variable(idx_sort))
        sort_lengths = lengths[idx_sort]

        sort_lengths = torch.tensor(sort_lengths.clone().cpu(), dtype=torch.int64)
        inputs_packed1 = nn.utils.rnn.pack_padded_sequence(sort_embed_input1,
                                                          sort_lengths,
                                                          batch_first=True)
        # process using RNN
        out_pack1, ht1 = self.gru1(inputs_packed1)
        raw_o = nn.utils.rnn.pad_packed_sequence(out_pack1, batch_first=True)
        raw_o = raw_o[0]
        raw_o = raw_o[idx_unsort]
        x = torch.tensor(np.linspace(0, raw_o.size(0), num=raw_o.size(0), endpoint=False), dtype=torch.long).to(self.device)
        y = lengths - 1
        outputs_last = raw_o[x, y]  # 2,2,6

        # ht1 = torch.transpose(ht1, 0, 1)[idx_unsort]
        # ht1 = torch.transpose(ht1, 0, 1)
        # out1 = self.fc1(ht1[-1])  # .squeeze()
        # out1 = self.fc1(outputs_last)

        return outputs_last  # K * hidden_size

class Discriminator(nn.Module):
    def __init__(self, config, device, dropout_p=0.2):
        super(Discriminator, self).__init__()
        self.device = device
        self.dropout_p = config.dropout
        self.input_size = config.num_product
        self.hidden_size = config.embedding_dim * 2
        self.max_basket_size = config.max_basket_size
        self.gru_hidden_size = config.embedding_dim

        self.fc1 = nn.Linear(self.gru_hidden_size, self.hidden_size)

        # TODO hidden_size
        self.judge_model1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(0),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_size, self.input_size)
        )
        self.judge_model2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(0),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_size, self.input_size)
        )

        self.dropout = nn.Dropout(dropout_p)
        # self.histroy = config.histroy
        # if self.histroy == 1:
        #     self.attn = nn.Linear(self.input_size, self.input_size)

    # profile
    def forward(self, item_embeddings1, history_record, target_tensor, input_seq_tensor=None):  # K * hidden_size  1*9963
        def hook_fn(grad):
            print(grad)

        item_embeddings1 = self.fc1(item_embeddings1)
        item_embeddings1 = self.dropout(item_embeddings1)
        judge1 = self.judge_model1(item_embeddings1)  # K*input_size
        judge2 = self.judge_model2(item_embeddings1)  # K*input_size
        judge = judge2 + judge1
        # judge = torch.softmax(judge2 + judge1, dim=-1)

        return judge  # K * n_items



