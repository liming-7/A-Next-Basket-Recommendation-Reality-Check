import numpy as np
import random
import sys
import os
import argparse
import logging
import json
from tqdm import tqdm
from time import time

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from basket_dataloader import *
from Explainablebasket import *
from utils import *
from pytorch_metric import *

class Trainer(object):

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.logger = logging.getLogger()
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.batch_size = config['batch_size']
        self.device = config['device']
        self.valid_metric = config['valid_metric']
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.train_file = config['train_file']
        self.tgt_file = config['tgt_file']
        self.data_config_file = config['data_config_file']
        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        saved_model_file = '{}-{}.pth'.format(self.config['model_name'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        # flags to record the training process
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -1
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self):
        r"""Init the Optimizer
        Returns:
            torch.optim: the optimizer
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, show_progress=False):
        r"""Train the model in an epoch
        Args:
            train_data : dataloader
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.
        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = self.model.calculate_loss
        total_loss = None

        # data prepare
        total_batch_cnt = int(train_data.__len__()/self.batch_size)
        for batch_id in range(total_batch_cnt):
            candidates = dict()
            data_train, tgt_train, repeat_data, explore_data = \
                train_data.get_batch_data(batch_id*self.batch_size, (batch_id+1)*self.batch_size)
            candidates['repeat']=repeat_data
            candidates['explore']=explore_data
            # train model according to the loss
            self.optimizer.zero_grad()
            losses = loss_func(data_train, tgt_train, candidates)
            # print(losses)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            # if self.clip_grad_norm:
            clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.optimizer.step()

        return total_loss #total loss is the loss sum of this epoch

    def _valid_epoch(self, valid_data, show_progress=False):
        r"""Valid the model with valid data
        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.
        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _save_checkpoint(self, epoch):
        r"""Store the model parameters information and training information.
        Args:
            epoch (int): the current epoch id
        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, self.saved_model_file)

    def resume_checkpoint(self, resume_file):
        r"""Load the model parameters information and training information.
        Args:
            resume_file (file): the checkpoint file
        """
        resume_file = str(resume_file)
        checkpoint = torch.load(resume_file)
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_step = checkpoint['cur_step']
        self.best_valid_score = checkpoint['best_valid_score']

        # load architecture params from checkpoint
        if checkpoint['config']['model'].lower() != self.config['model'].lower():
            self.logger.warning(
                'Architecture configuration given in config file is different from that of checkpoint. '
                'This may yield an exception while state_dict is being loaded.'
            )
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        print(message_output)

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        # generate output for logger
        des = 4
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = 'train_loss%d: %.' + str(des) + 'f'
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += 'train loss:' + des % losses
        return train_loss_output + ']'

    def fit(self, basket_dataset, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.
        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.
        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        first_score, first_result = self._valid_epoch(valid_data, show_progress=show_progress)
        print("Start valid:", first_score, first_result)
        sys.stdout.flush()
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(basket_dataset, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                print(train_loss_output)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = 'Saving current: %s' % self.saved_model_file
                    if verbose:
                        print(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step
                    # bigger=self.valid_metric_bigger #whether the bigger the better
                )
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                if verbose:
                    print(valid_score_output)
                    print(valid_result_output)
                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = 'Saving current best: %s' % self.saved_model_file
                        if verbose:
                            print(update_output)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        print(stop_output)
                    break
            sys.stdout.flush()

        return self.best_valid_score, self.best_valid_result

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.
        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            print(message_output)

        self.model.eval()

        candidates = dict()
        data_train, tgt_train, repeat_data, explore_data = eval_data.get_batch_data(0, eval_data.__len__())
        candidates['repeat'] = repeat_data
        candidates['explore'] = explore_data
        scores = self.model.forward(data_train, candidates)
        tgt_labels = get_label_tensor(tgt_train, self.device, max_index=eval_data.total_num)
        result = dict()
        result['recall10'] = recall(scores, tgt_labels, top_k=10)
        result['recall20'] = recall(scores, tgt_labels, top_k=20)
        result['recall40'] = recall(scores, tgt_labels, top_k=40)
        result['ndcg10'] = ndcg(scores, tgt_labels, top_k=10)
        result['ndcg20'] = ndcg(scores, tgt_labels, top_k=20)
        result['ndcg40'] = ndcg(scores, tgt_labels, top_k=40)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    parser.add_argument('--fold_id', type=int, default=0, help='x')
    parser.add_argument('--loss_mode', type=int, default=0, help='x')
    parser.add_argument('--attention', type=int, default=0)
    args = parser.parse_args()
    dataset = args.dataset
    fold_id = args.fold_id
    loss_mode = args.loss_mode
    attention = args.attention

    train_setting = args.dataset
    with open(dataset+'conf.json', 'r') as f:
        train_config = json.load(f)
    train_config['valid_metric'] = "recall20"
    train_config['loss_mode'] = loss_mode
    train_config['data_config_file'] = train_config['data_config_file']+str(fold_id)+'.json'
    train_config['attention'] = args.attention
    train_config['model_name'] = f"{dataset}-recall20-{loss_mode}-{fold_id}-{attention}-"

    with open(train_config['data_config_file'], 'r') as f:
        dataset_config = json.load(f)
    print(train_config['data_config_file'])
    sys.stdout.flush()
    train_config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    basket_dataset = BasketDataset(train_config, mode='train')
    validate_dataset = BasketDataset(train_config, mode='val')
    model = NBRNet(train_config, dataset_config)
    print('Device:', train_config['device'])
    sys.stdout.flush()
    model.to(train_config['device'])
    trainer = Trainer(train_config, model)
    trainer.fit(basket_dataset, valid_data=validate_dataset)
    test_dataset = BasketDataset(train_config, mode='test')
    results = trainer.evaluate(test_dataset, load_best_model=True)
    print(results)
    sys.stdout.flush()