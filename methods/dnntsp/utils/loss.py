import random

import torch
import torch.nn as nn
import itertools
import sys
sys.path.append("..")


class BPRLoss(nn.Module):

    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, predict, truth):
        """
        Args:
            predict: (batch_size, items_total) / (batch_size, baskets_num, item_total)
            truth: (batch_size, items_total) / (batch_size, baskets_num, item_total)
        Returns:
            output: tensor
        """
        result = self.batch_bpr_loss(predict, truth)

        return result

    def batch_bpr_loss(self, predict, truth):
        """
            Args:
                predict: (batch_size, items_total)
                truth: (batch_size, items_total)
            Returns:
                output: tensor
        """
        items_total = truth.shape[1]
        nll = 0
        for user, predictUser in zip(truth, predict):
            pos_idx = torch.tensor(user, dtype=torch.uint8)
            preUser = predictUser[pos_idx]
            non_zero_list = list(itertools.chain.from_iterable(torch.nonzero(user)))
            random_list = list(set(range(0, items_total)) - set(non_zero_list))
            random.shuffle(random_list)
            neg_idx = torch.tensor(random_list[:len(preUser)])
            score = preUser - predictUser[neg_idx]
            nll += - torch.mean(torch.nn.LogSigmoid()(score))
        return nll


class WeightMSELoss(nn.Module):

    def __init__(self, weights=None):
        """
        Args:
            weights: tensor, (items_total, )
        """
        super(WeightMSELoss, self).__init__()
        self.weights = weights
        if weights is not None:
            self.weights = torch.sqrt(weights)
        self.mse_loss = nn.MSELoss(reduction='sum')

    def forward(self, predict, truth):
        """
        Args:
            predict: tenor, (batch_size, items_total)
            truth: tensor, (batch_size, items_total)
        Returns:
            output: tensor
        """
        # predict = torch.softmax(predict, dim=-1)
        predict = torch.sigmoid(predict)
        truth = truth.float()
        # print(predict.device)
        # print(truth.device)
        if self.weights is not None:
            self.weights = self.weights.to(truth.device)
            predict = predict * self.weights
            truth = truth * self.weights

        loss = self.mse_loss(predict, truth)
        return loss

class W_multilabel(nn.Module):

    def __init__(self, weight):
        self.weights = weight

    def forward(self, predict, truth):
        predict = torch.sigmoid(predict)
        truth = truth.float()
        assert len(self.weights) == 1
        self.weights = self.weights.to(truth.device)
        batch_size = predict.size(0)
        w_loss = 0
        for ind in range(batch_size):
            c_loss = self.weights * (truth[ind] * torch.log(predict[ind])) + \
                       ((1 - truth[ind]) * torch.log(1 - predict[ind]))
            w_loss += torch.neg(torch.sum(c_loss))
        return w_loss
