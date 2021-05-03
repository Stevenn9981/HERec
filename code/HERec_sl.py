#!/usr/bin/python
# encoding=utf-8
import collections

import numpy as np
import time
import random
from math import sqrt, fabs, log
import sys

import torch


class HNERec:
    def __init__(self, unum, inum, ratedim, userdim, itemdim, user_metapaths, item_metapaths, trainfile, testfile,
                 steps, delta, beta_e, beta_h, beta_p, beta_w, beta_b, reg_u, reg_v):
        self.unum = unum
        self.inum = inum
        self.ratedim = ratedim
        self.userdim = userdim
        self.itemdim = itemdim
        self.steps = steps
        self.delta = delta
        self.beta_e = beta_e
        self.beta_h = beta_h
        self.beta_p = beta_p
        self.beta_w = beta_w
        self.beta_b = beta_b
        self.reg_u = reg_u
        self.reg_v = reg_v

        self.user_metapathnum = len(user_metapaths)
        self.item_metapathnum = len(item_metapaths)

        self.train_user_dict = collections.defaultdict(list)
        self.test_user_dict = collections.defaultdict(list)

        self.X, self.user_metapathdims = self.load_embedding(user_metapaths, unum)
        print('Load user embeddings finished.')

        self.Y, self.item_metapathdims = self.load_embedding(item_metapaths, inum)
        print('Load user embeddings finished.')

        self.R, self.T, self.ba = self.load_rating(trainfile, testfile)
        print('Load rating finished.')
        print('train size : ', len(self.R))
        print('test size : ', len(self.T))

        self.initialize()
        self.recommend()

    def load_embedding(self, metapaths, num):
        X = np.zeros((num, len(metapaths), 128))
        metapathdims = []

        ctn = 0
        for metapath in metapaths:
            sourcefile = '../data/embeddings/' + metapath
            # print sourcefile
            with open(sourcefile) as infile:

                k = int(infile.readline().strip().split(' ')[1])
                metapathdims.append(k)

                n = 0
                for line in infile.readlines():
                    n += 1
                    arr = line.strip().split(' ')
                    i = int(arr[0]) - 1
                    for j in range(k):
                        X[i][ctn][j] = float(arr[j + 1])
                print('metapath ', metapath, 'numbers ', n)
            ctn += 1
        return X, metapathdims

    def load_rating(self, trainfile, testfile):
        R_train = []
        R_test = []
        ba = 0.0
        n = 0
        user_test_dict = dict()
        with open(trainfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                R_train.append([int(user) - 1, int(item) - 1, int(rating)])
                ba += int(rating)
                n += 1
                self.train_user_dict[int(user) - 1].append(int(item) - 1)
        ba = ba / n
        ba = 0
        with open(testfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                R_test.append([int(user) - 1, int(item) - 1, int(rating)])
                self.test_user_dict[int(user) - 1].append(int(item) - 1)

        return R_train, R_test, ba

    def initialize(self):
        self.E = np.random.randn(self.unum, self.itemdim) * 0.1
        self.H = np.random.randn(self.inum, self.userdim) * 0.1
        self.U = np.random.randn(self.unum, self.ratedim) * 0.1
        self.V = np.random.randn(self.inum, self.ratedim) * 0.1

        self.pu = np.ones((self.unum, self.user_metapathnum)) * 1.0 / self.user_metapathnum
        self.pv = np.ones((self.inum, self.item_metapathnum)) * 1.0 / self.item_metapathnum

        self.Wu = {}
        self.bu = {}
        for k in range(self.user_metapathnum):
            self.Wu[k] = np.random.randn(self.userdim, self.user_metapathdims[k]) * 0.1
            self.bu[k] = np.random.randn(self.userdim) * 0.1

        self.Wv = {}
        self.bv = {}
        for k in range(self.item_metapathnum):
            self.Wv[k] = np.random.randn(self.itemdim, self.item_metapathdims[k]) * 0.1
            self.bv[k] = np.random.randn(self.itemdim) * 0.1

    def cal_u(self, i):
        ui = np.zeros(self.userdim)
        for k in range(self.user_metapathnum):
            ui += self.pu[i][k] * (self.Wu[k].dot(self.X[i][k]) + self.bu[k])
        return ui

    def cal_v(self, j):
        vj = np.zeros(self.itemdim)
        for k in range(self.item_metapathnum):
            vj += self.pv[j][k] * (self.Wv[k].dot(self.Y[j][k]) + self.bv[k])
        return vj

    def get_rating(self, i, j):
        ui = self.cal_u(i)
        vj = self.cal_v(j)
        return self.U[i, :].dot(self.V[j, :]) + self.reg_u * ui.dot(self.H[j, :]) + self.reg_v * self.E[i, :].dot(vj)

    def maermse(self):
        m = 0.0
        mae = 0.0
        rmse = 0.0
        n = 0
        for t in self.T:
            n += 1
            i = t[0]
            j = t[1]
            r = t[2]
            r_p = self.get_rating(i, j)

            if r_p > 5: r_p = 5
            if r_p < 1: r_p = 1
            m = fabs(r_p - r)
            mae += m
            rmse += m * m
        mae = mae * 1.0 / n
        rmse = sqrt(rmse * 1.0 / n)
        return mae, rmse

    def recommend(self):
        mae = []
        rmse = []
        ndcg = []
        starttime = time.time()
        perror = 99999
        cerror = 9999
        n = len(self.R)

        for step in range(steps):
            total_error = 0.0
            for t in self.R:
                i = t[0]
                j = t[1]
                rij = t[2]

                rij_t = self.get_rating(i, j)
                eij = rij - rij_t
                total_error += eij * eij

                U_g = -eij * self.V[j, :] + self.beta_e * self.U[i, :]
                V_g = -eij * self.U[i, :] + self.beta_h * self.V[j, :]

                self.U[i, :] -= self.delta * U_g
                self.V[j, :] -= self.delta * V_g

                ui = self.cal_u(i)
                for k in range(self.user_metapathnum):
                    pu_g = self.reg_u * -eij * self.H[j, :].dot(
                        self.Wu[k].dot(self.X[i][k]) + self.bu[k]) + self.beta_p * self.pu[i][k]
                    Wu_g = self.reg_u * -eij * self.pu[i][k] * np.array([self.H[j, :]]).T.dot(
                        np.array([self.X[i][k]])) + self.beta_w * self.Wu[k]
                    bu_g = self.reg_u * -eij * self.pu[i][k] * self.H[j, :] + self.beta_b * self.bu[k]

                    # self.pu[i][k] -= 0.1 * self.delta * pu_g
                    self.Wu[k] -= 0.1 * self.delta * Wu_g
                    self.bu[k] -= 0.1 * self.delta * bu_g

                H_g = self.reg_u * -eij * ui + self.beta_h * self.H[j, :]
                self.H[j, :] -= self.delta * H_g

                vj = self.cal_v(j)
                for k in range(self.item_metapathnum):
                    pv_g = self.reg_v * -eij * self.E[i, :].dot(
                        self.Wv[k].dot(self.Y[j][k]) + self.bv[k]) + self.beta_p * self.pv[j][k]
                    Wv_g = self.reg_v * -eij * self.pv[j][k] * np.array([self.E[i, :]]).T.dot(
                        np.array([self.Y[j][k]])) + self.beta_w * self.Wv[k]
                    bv_g = self.reg_v * -eij * self.pv[j][k] * self.E[i, :] + self.beta_b * self.bv[k]

                    # self.pv[j][k] -= 0.1 * self.delta * pv_g
                    self.Wv[k] -= 0.1 * self.delta * Wv_g
                    self.bv[k] -= 0.1 * self.delta * bv_g

                E_g = self.reg_v * -eij * vj + 0.01 * self.E[i, :]
                self.E[i, :] -= self.delta * E_g
            perror = cerror
            cerror = total_error / n

            self.delta = self.delta * 0.93
            if (abs(perror - cerror) < 0.0001):
                break
            # print 'step ', step, 'crror : ', sqrt(cerror)
            # MAE, RMSE = self.maermse()
            # mae.append(MAE)
            # rmse.append(RMSE)
            NDCG = self.test_batch()
            ndcg.append(NDCG)
            print('NDCG@10: ', NDCG)
            # print 'MAE, RMSE ', MAE, RMSE
            endtime = time.time()
            print('time: ', endtime - starttime)
        # print('MAE: ', min(mae), ' RMSE: ', min(rmse))
        print('NDCG@10: ', min(ndcg), "index: ", ndcg.index(min(ndcg)))

    def cal_us(self):
        us = np.zeros((self.unum, self.userdim))
        for k in range(self.user_metapathnum):
            us += self.pu[:, k].reshape(self.unum, 1) * (self.Wu[k].dot(self.X[:, k].T).T + self.bu[k])
        return us

    def cal_vs(self):
        vs = np.zeros((self.inum, self.itemdim))
        for k in range(self.item_metapathnum):
            vs += self.pv[:, k].reshape(self.inum, 1) * (self.Wv[k].dot(self.Y[:, k].T).T + self.bv[k])
        return vs

    def get_ratings(self):
        us = self.cal_us()
        vs = self.cal_vs()
        return self.U.dot(self.V.T) + self.reg_u * us.dot(self.H.T) + self.reg_v * self.E.dot(vs.T)

    def test_batch(self):
        user_ids = list(self.test_user_dict.keys())
        user_ids_batch = user_ids[:]
        neg_dict = collections.defaultdict(list)

        for u in user_ids_batch:
            for _ in self.test_user_dict[u]:
                nl = self.sample_neg_items_for_u_test(self.train_user_dict, self.test_user_dict, u, 100)
                neg_dict[u].extend(nl)

        pos_logits = torch.tensor([])
        neg_logits = torch.tensor([])

        scores = self.get_ratings()

        for u in user_ids_batch:
            pos_logits = torch.cat([pos_logits, torch.from_numpy(scores[u][self.test_user_dict[u]])])
            neg_logits = torch.cat([neg_logits, torch.unsqueeze(torch.from_numpy(scores[u][neg_dict[u]]), 1)])

        HR1, HR3, HR20, HR50, MRR10, MRR20, MRR50, NDCG10, NDCG20, NDCG50 = self.metrics(pos_logits, neg_logits)
        print("HR1 : %.4f, HR3 : %.4f, HR20 : %.4f, HR50 : %.4f, MRR10 : %.4f, MRR20 : %.4f, MRR50 : %.4f, "
                     "NDCG10 : %.4f, NDCG20 : %.4f, NDCG50 : %.4f" % (HR1, HR3, HR20, HR50, MRR10.item(), MRR20.item(),
                                                                      MRR50.item(), NDCG10.item(), NDCG20.item(),
                                                                      NDCG50.item()))

        return NDCG10.cpu().item()

    def metrics(self, batch_pos, batch_nega):
        hit_num1 = 0.0
        hit_num3 = 0.0
        hit_num20 = 0.0
        hit_num50 = 0.0
        mrr_accu10 = torch.tensor(0)
        mrr_accu20 = torch.tensor(0)
        mrr_accu50 = torch.tensor(0)
        ndcg_accu10 = torch.tensor(0)
        ndcg_accu20 = torch.tensor(0)
        ndcg_accu50 = torch.tensor(0)


        batch_neg_of_user = torch.split(batch_nega, 100, dim=0)

        for i in range(batch_pos.shape[0]):
            pre_rank_tensor = torch.cat((batch_pos[i].view(1, 1), batch_neg_of_user[i]), dim=0)
            _, indices = torch.topk(pre_rank_tensor, k=pre_rank_tensor.shape[0], dim=0)
            rank = torch.squeeze((indices == 0).nonzero())
            rank = rank[0]
            if rank < 50:
                ndcg_accu50 = ndcg_accu50 + torch.log(torch.tensor([2.0])) / torch.log((rank + 2).type(torch.float32))
                mrr_accu50 = mrr_accu50 + 1 / (rank + 1).type(torch.float32)
                hit_num50 = hit_num50 + 1
            if rank < 20:
                ndcg_accu20 = ndcg_accu20 + torch.log(torch.tensor([2.0])) / torch.log((rank + 2).type(torch.float32))
                mrr_accu20 = mrr_accu20 + 1 / (rank + 1).type(torch.float32)
                hit_num20 = hit_num20 + 1
            if rank < 10:
                ndcg_accu10 = ndcg_accu10 + torch.log(torch.tensor([2.0])) / torch.log((rank + 2).type(torch.float32))
            if rank < 10:
                mrr_accu10 = mrr_accu10 + 1 / (rank + 1).type(torch.float32)
            if rank < 3:
                hit_num3 = hit_num3 + 1
            if rank < 1:
                hit_num1 = hit_num1 + 1
        return hit_num1 / batch_pos.shape[0], hit_num3 / batch_pos.shape[0], hit_num20 / batch_pos.shape[0], hit_num50 / \
               batch_pos.shape[0], mrr_accu10 / batch_pos.shape[0], mrr_accu20 / batch_pos.shape[0], mrr_accu50 / \
               batch_pos.shape[0], \
               ndcg_accu10 / batch_pos.shape[0], ndcg_accu20 / batch_pos.shape[0], ndcg_accu50 / batch_pos.shape[0]

    def sample_neg_items_for_u_test(self, user_dict, test_dict, user_id, n_sample_neg_items):
        pos_items = user_dict[user_id]
        pos_items_2 = test_dict[user_id]

        sample_neg_items = []
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break
            neg_item_id = np.random.randint(low=0, high=14284, size=1)[0]
            if neg_item_id not in pos_items and neg_item_id not in pos_items_2 and neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return sample_neg_items

if __name__ == "__main__":
    unum = 16239
    inum = 14284
    ratedim = 10
    userdim = 30
    itemdim = 10
    train_rate = 0.8  # float(sys.argv[1])

    user_metapaths = ['ubu', 'ubcibu', 'ubcabu']
    item_metapaths = ['bub', 'bcib', 'bcab']

    for i in range(len(user_metapaths)):
        user_metapaths[i] += '_' + str(train_rate) + '.embedding'
    for i in range(len(item_metapaths)):
        item_metapaths[i] += '_' + str(train_rate) + '.embedding'
    # user_metapaths = ['ubu_' + str(train_rate) + '.embedding', 'ubcibu_'+str(train_rate)+'.embedding', 'ubcabu_'+str(train_rate)+'.embedding']

    # item_metapaths = ['bub_'+str(train_rate)+'.embedding', 'bcib.embedding', 'bcab.embedding']
    trainfile = '../data/ub_' + str(train_rate) + '.train'
    testfile = '../data/ub_' + str(train_rate) + '.test'
    steps = 100
    delta = 0.02
    beta_e = 0.1
    beta_h = 0.1
    beta_p = 2
    beta_w = 0.1
    beta_b = 0.01
    reg_u = 1.0
    reg_v = 1.0
    print('train_rate: ', train_rate)
    print('ratedim: ', ratedim, ' userdim: ', userdim, ' itemdim: ', itemdim)
    print('max_steps: ', steps)
    print('delta: ', delta, 'beta_e: ', beta_e, 'beta_h: ', beta_h, 'beta_p: ', beta_p, 'beta_w: ', beta_w, 'beta_b',
          beta_b, 'reg_u', reg_u, 'reg_v', reg_v)

    HNERec(unum, inum, ratedim, userdim, itemdim, user_metapaths, item_metapaths, trainfile, testfile, steps, delta,
           beta_e, beta_h, beta_p, beta_w, beta_b, reg_u, reg_v)
