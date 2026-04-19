from __future__ import absolute_import, print_function

import os
import random
import time
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from reid.evaluaters import accuracy
from reid.loss import (CrossEntropyLabelSmooth, SoftEntropy, SoftTripletLoss,
                       TripletLoss)
from reid.tools import AverageMeter, time_now

# for veri: step for 2 iters
class GCCTrainer(object):
    def __init__(self, model,  gat, memory_strategy, num_cluster=500, alpha=0.999, args=None):
        super(GCCTrainer, self).__init__()
        self.model = model
        self.num_cluster = num_cluster
        self.gat = gat
        self.alpha = alpha
        self.memory_strategy = memory_strategy

        self.momentum1 = args.momentum1
        self.momentum2 = args.momentum2

        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        print('train v2--weight')

    def train(self, epoch, data_loader, optimizer, memorybank,
              print_freq=1, train_iters=200, loss_weight = 0.1, beta = 0.0, k=16):

        self.model.train()

        precision = AverageMeter()
        losses = AverageMeter()

        for iter_idx in range(train_iters):
            inputs, fname, targets, cid = data_loader.next()
            inputs = inputs.cuda()
            targets = targets.cuda()

            batch_size = inputs.shape[0]
            labels = targets.tolist()

            # forward
            f, _, fbn = self.model(inputs)

            p = torch.mm(fbn, memorybank.t()) / beta


            # cross entropy for classification
            loss_ce = F.cross_entropy(p, targets)

            loss = loss_ce

            loss.backward()
            
            if (iter_idx + 1)% 2 == 0:
                optimizer.step()
                optimizer.zero_grad()

            prec, = accuracy(p.data, targets.data)
            precision.update(prec[0].item())

            losses.update(loss.item())

            if epoch >= 0:
                self._update_memorybank(memorybank, fbn, targets, self.memory_strategy)
                memorybank.detach_()

            if ((iter_idx + 1) % print_freq == 0) or (iter_idx == 0):
                prec = str(round(precision.avg*100, 4))
                print(time_now() +
                      '[Epoch:{:03d}] [{:03d}/{:03d}] | '
                      'Loss: {:2.3f} | '
                      'Acc: {:s}'
                      .format(epoch, iter_idx + 1, len(data_loader),
                              losses.avg,
                              prec))


    def _update_memorybank(self, memorybank, batch_features, batch_labels, memory_strategy):

        if 'base' in memory_strategy:
            for feature, label in zip(batch_features, batch_labels.tolist()):
                memorybank[label] = (1-self.momentum1)* memorybank[label] + self.momentum1 * feature
                memorybank[label] /= memorybank[label].norm()
                # print(f'\r updateing{label}',end='-----')
            return 0

        N = memorybank.shape[0] ## number of classes

        C = memorybank.mean(0)  ## center of memory bank

        classes_features = defaultdict(list)
        for feature, label in zip(batch_features, batch_labels.tolist()):
            classes_features[label].append(feature)
        
        for index, features in classes_features.items():
            # 'mean-far-easy-random-update-replace'

            if 'mean' in memory_strategy: ## use all samples
                # f = sum(features) / len(features)  ## sample for pulling
                features = torch.stack(features)
                similarities = torch.mm(features, memorybank[index].unsqueeze(1)).squeeze()
                g1 = 0
                for i in range(len(features)):
                    f = features[i]
                    g1 += (1 - similarities[i]) * (memorybank[index] - f)
                g1 /= len(features)

                similarities = torch.mm(memorybank, memorybank[index].unsqueeze(1)).squeeze()                
                g2 = 0
                for i in range(N):
                    if i == index:
                        continue
                    g2 += (1 + similarities[i]) * (memorybank[i] + memorybank[index])
                g2 /= N
            
            elif 'far' in memory_strategy:  ## use the hardest sample
                features = torch.stack(features)
                similarities = torch.mm(features, memorybank[index].unsqueeze(1)).squeeze()
                f = features[torch.argmin(similarities)]  ## sample for pulling
                g1 = (1 - memorybank[index].dot(f)) * (memorybank[index] - f)
                
                similarities = torch.mm(memorybank, memorybank[index].unsqueeze(1)).squeeze()
                # input(similarities[index])
                similarities[index] = 0
                c = memorybank[torch.argmax(similarities)]
                g2 = (1 + memorybank[index].dot(c)) * (memorybank[index] + c)
            
            elif 'easy' in memory_strategy:
                features = torch.stack(features)
                similarities = torch.mm(features, memorybank[index].unsqueeze(1)).squeeze()
                f = features[torch.argmax(similarities)]  ## sample for pulling
                g1 = (1 - memorybank[index].dot(f)) * (memorybank[index] - f)

                similarities = torch.mm(memorybank, memorybank[index].unsqueeze(1)).squeeze()
                # input(similarities[index])
                similarities[index] = 0
                c = memorybank[torch.argmin(similarities)]
                g2 = (1 + memorybank[index].dot(c)) * (memorybank[index] + c)

            elif 'random' in memory_strategy:
                f = features[random.randint(0, len(features)-1)]
                c = memorybank[(random.randint(1, N-1)+index) % N]
                g1 = (1 - memorybank[index].dot(f)) * (memorybank[index] - f)
                g2 = (1 + memorybank[index].dot(c)) * (memorybank[index] + c)
                
            else:
                print('unknown memory update strategy:', memory_strategy)
                import os
                os._exist(0)

            if 'update' in memory_strategy:
                # memorybank[index] = (1 - self.momentum1 - self.momentum2*(N-2)/N ) * memorybank[index] \
                #                     + self.momentum1 * f - self.momentum2 * C
                memorybank[index] -= (self.momentum1 * g1 + self.momentum2 * g2)
                memorybank[index] = F.normalize(memorybank[index].unsqueeze(0))[0]
            elif 'replace' in memory_strategy:
                memorybank[index] = F.normalize(f.unsqueeze(0))[0]
            else:
                print('unknown memory update strategy:', memory_strategy)
                import os
                os._exist(0)
                