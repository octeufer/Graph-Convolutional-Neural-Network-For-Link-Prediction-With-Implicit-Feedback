from copy import deepcopy
import sys
sys.path.append('..')

import os
import time
import fire
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import GCMC
from data import MovieLens, ASOS

import sklearn.metrics as metrics

class Trainer:
    def __init__(self,
                data_name = 'ASOS',
                valid_ratio = 0.1,
                test_ratio = 0.1
                ):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = ASOS(name = data_name,
                                test_ratio = test_ratio,
                                valid_ratio = valid_ratio,
                                device = device)

    def train(self,
            n_layers = 1,
            hidden_feats_dim = 500,
            out_feats_dim = 75,
            agg = 'stack',
            drop_out = 0.7,
            activation = 'leaky',
            n_basis = 2,
            lr = 0.01,
            iteration = 2000,
            log_interval = 1,
            early_stopping = 100,
            lr_intialize_step = 50,
            lr_decay = 0.5,
            train_min_lr = 0.001
            ):

        logging.basicConfig(filename='./train.log')

        model = GCMC(n_layers = n_layers,
                    edge_types = self.dataset.possible_rating_values,
                    user_feats_dim = self.dataset.user_feature_shape[1],
                    item_feats_dim = self.dataset.movie_feature_shape[1],
                    hidden_feats_dim = hidden_feats_dim,
                    out_feats_dim = out_feats_dim,
                    agg = agg,
                    drop_out = drop_out,
                    activation = activation,
                    n_basis = n_basis)
        print(model)

        device = self.dataset._device
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = lr)
        possible_edge_types = torch.FloatTensor(self.dataset.possible_rating_values).unsqueeze(0).to(device)

        train_gt_labels = self.dataset.train_labels
        train_gt_ratings = self.dataset.train_truths

        labels_cpu = deepcopy(train_gt_labels.data)
        labels_cpu = labels_cpu.detach().cpu().numpy()

        best_valid_auc = 0
        best_valid_precision = 0
        best_valid_recall = 0
        no_better_valid = 0
        best_iter = -1
        count_rmse = 0
        count_num = 0
        count_loss = 0

        self.dataset.train_enc_graph = self.dataset.train_enc_graph.int().to(device)
        self.dataset.train_dec_graph = self.dataset.train_dec_graph.int().to(device)
        self.dataset.valid_enc_graph = self.dataset.train_enc_graph
        self.dataset.valid_dec_graph = self.dataset.valid_dec_graph.int().to(device)
        self.dataset.test_enc_graph = self.dataset.test_enc_graph.int().to(device)
        self.dataset.test_dec_graph = self.dataset.test_dec_graph.int().to(device)

        print(f"Start training on {device}...")
        for iter_idx in range(iteration):
            model.train()
            logits = model(self.dataset.train_enc_graph, self.dataset.train_dec_graph,
                            self.dataset.user_feature, self.dataset.movie_feature)
            loss = criterion(logits, train_gt_labels).mean()
            count_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # pred_{i,j} = \sum_{r = 1} r * P(link_{i,j} = r)
            pred_ratings = (torch.softmax(logits, dim=1) * possible_edge_types).sum(dim=1)
            
            # mse = ((pred_ratings - train_gt_ratings) ** 2).sum()
            # rmse = mse.pow(1/2)
            # count_rmse += rmse.item()
            # count_num += pred_ratings.shape[0]
            
            auc = metrics.roc_auc_score(labels_cpu,pred_ratings.detach().cpu().numpy())
            precision = metrics.precision_score(labels_cpu, ((pred_ratings>0.5)*1).detach().cpu().numpy())
            recall = metrics.recall_score(labels_cpu, ((pred_ratings>0.5)*1).detach().cpu().numpy())

            if iter_idx and iter_idx % log_interval == 0:
                log = f"[{iter_idx}/{iteration}-iter] | [train] loss : {count_loss/iter_idx:.4f}, precision : {precision:.4f}, auc : {auc:.4f}, recall : {recall:.4f}"
                count_rmse, count_num = 0, 0

            if iter_idx and iter_idx % (log_interval*10) == 0:
                valid_auc, valid_precision, valid_recall = self.evaluate(model, self.dataset, possible_edge_types, data_type = 'valid')
                log += f" | [valid] precision : {valid_precision:.4f}, auc : {valid_auc:.4f}, recall : {valid_recall:.4f}"

                if valid_auc > best_valid_auc:
                    best_valid_auc = valid_auc
                    best_valid_precision = valid_precision
                    best_valid_recall = valid_recall
                    no_better_valid = 0
                    best_iter = iter_idx
                    best_test_auc, best_test_precision, best_test_recall = self.evaluate(model, self.dataset, possible_edge_types, data_type = 'test')
                    log += f" | [test] precision : {best_test_precision:.4f}, auc : {best_test_auc:.4f}, recall : {best_test_recall:.4f}"

                    torch.save(model, './model.pt')

                else:
                    no_better_valid += 1
                    if no_better_valid > early_stopping:
                        logging.info("Early stopping threshold reached. Stop training.")
                        break
                    if no_better_valid > lr_intialize_step:
                        new_lr = max(lr * lr_decay, train_min_lr)
                        if new_lr < lr:
                            lr = new_lr
                            logging.info("\tChange the LR to %g" % new_lr)
                            for p in optimizer.param_groups:
                                p['lr'] = lr
                            no_better_valid = 0

            if iter_idx and iter_idx  % log_interval == 0:
                print(log)

        print(f'[END] Best Iter : {best_iter} Best Valid AUC : {best_valid_auc:.4f}, Best Valid Precision : {best_valid_precision:.4f}, \
            Best Valid Recall : {best_valid_recall:.4f}, \r\n Best Test AUC : {best_test_auc:.4f}, Best Test Precision : {best_test_precision:.4f}, Best Test Recall : {best_test_recall:.4f}')

    def evaluate(self, model, dataset, possible_edge_types, data_type = 'valid'):
        if data_type == "valid":
            rating_values = dataset.valid_truths
            enc_graph = dataset.valid_enc_graph
            dec_graph = dataset.valid_dec_graph
        elif data_type == "test":
            rating_values = dataset.test_truths
            enc_graph = dataset.test_enc_graph
            dec_graph = dataset.test_dec_graph

        labels_cpu = rating_values.detach().cpu().numpy()

        model.eval()
        with torch.no_grad():
            logits = model(enc_graph, dec_graph,
                            dataset.user_feature, dataset.movie_feature)
            pred_ratings = (torch.softmax(logits, dim=1) * possible_edge_types).sum(dim=1)
            # mse = ((pred_ratings - rating_values) ** 2.).mean().item()
            # rmse = np.sqrt(mse)

            auc = metrics.roc_auc_score(labels_cpu,pred_ratings.detach().cpu().numpy())
            precision = metrics.precision_score(labels_cpu, ((pred_ratings>0.5)*1).detach().cpu().numpy())
            recall = metrics.recall_score(labels_cpu, ((pred_ratings>0.5)*1).detach().cpu().numpy())
            
        return auc, precision, recall

if __name__ == '__main__':
    np.random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
    fire.Fire(Trainer)
