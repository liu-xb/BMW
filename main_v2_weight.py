from __future__ import absolute_import, print_function

import argparse
import collections
import datetime
import os
import os.path as osp
import random
import sys
import gc

import numpy as np
import torch
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.preprocessing import normalize
from torch import nn
from torch.backends import cudnn

from reid.data import get_dataset, get_test_loader, get_train_loader
from reid.evaluaters import Evaluator, extract_features
from reid.gcctrain_v2_weight import GCCTrainer
from reid.model import create_model
from reid.rerank import compute_jaccard_distance, re_ranking
from reid.tools import (Logger, check_gpu, load_checkpoint, save_model,
                        time_now, to_torch)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def main(args):
    BEGIN_TIME = datetime.datetime.now()
    torch.set_printoptions(precision=4, threshold=100)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    cudnn.benchmark = True

    args.logs += f'{args.batch_size}b-{args.iters}iter-{args.step}step-{args.eps}eps-{args.epochs}epochs-{args.beta}beta-{args.memory_strategy}-{args.momentum1}-{args.momentum2}'
    if os.path.exists(args.logs):
        print(f'there already is {args.logs}'
               'Press \'y\' or 1 to remove the file and continue. Press other keys to exit.')
        temp_input = input()
        if temp_input == 'y' or temp_input == 1 or temp_input == '1':
            print('\rContinue training in '+args.logs)
        else:
            return
    check_gpu(args.gpus, need_memory = 23000, wait_level = 2)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    sys.stdout = Logger(osp.join(args.logs, 'train.log'))

    print(time_now())
    print("==========")
    args_keys = list(args.__dict__.keys())
    args_keys.sort()
    for k in args_keys:
        print(f'{k} : {args.__dict__[k]}')
    print("==========")

    dataset_target = get_dataset(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, int(args.batch_size/2), args.workers)

    model, _, gat = create_model(args)
    # # Optimizer
    # # if ((epoch-args.start_epoch+1) % args.step) == 0:
    # #     args.lr /= 10.
    # params = []
    # for key, value in model.named_parameters():
    #     if not value.requires_grad:
    #         continue
    #     params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    # optimizer = torch.optim.Adam(params)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)

    # print('learning rate: ', args.lr)
    evaluator = Evaluator(model)
    train_data_cache = None #[]

        
    for epoch in range(args.start_epoch, args.epochs):
        torch.cuda.empty_cache()
        cluster_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size,
                                        args.workers, testset=sorted(dataset_target.train))
        features, _ = extract_features(model, cluster_loader, print_freq=100, catch_data = train_data_cache)

        # input(features)
        # features = torch.stack(list(features.values()))
        features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset_target.train)], 0)


        # flag = int((epoch - args.start_epoch) / args.step) % 2
        flag = False
        # flag = epoch%2
        if  flag:
            print(f'\n {time_now()} K-means clusters into {args.num_clusters} classes \n')
            km = MiniBatchKMeans(n_clusters=args.num_clusters, max_iter=100, batch_size=2560, init_size=1500).fit(features)
            num_ids = args.num_clusters
            cluster_centers = km.cluster_centers_

            # change pseudo labels
            target_label = km.labels_
            for i in range(len(dataset_target.train)):
                dataset_target.train[i] = list(dataset_target.train[i])
                dataset_target.train[i][1] = int(target_label[i])
                dataset_target.train[i] = tuple(dataset_target.train[i])

            train_loader = get_train_loader(dataset_target, args.height, args.width, args.batch_size,
                                            args.workers, args.num_instances, args.iters)
        else:
            rerank_dist = compute_jaccard_distance(features)
            # rerank_dist = re_ranking(features)
            print(f'{time_now()} start dbscan clustering')
            # tri_mat = np.triu(rerank_dist,1)
            # tri_mat = tri_mat[np.nonzero(tri_mat)]
            # tri_mat = np.sort(tri_mat,axis=None)
            # top_num = np.round(1.6e-3*tri_mat.size).astype(int)
            # eps = tri_mat[:top_num].mean()
            eps = args.eps # or 0.4
            cluster = DBSCAN(eps=eps,min_samples=4, metric='precomputed', n_jobs=-1)
            labels = cluster.fit_predict(rerank_dist)
            num_ids = len(set(labels)) - (1 if -1 in labels else 0)
            print(f'{time_now()}\n Clustered into {num_ids} classes \n')

            new_dataset = []
            cluster_centers = collections.defaultdict(list)
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_target.train), labels)):
                if label==-1:
                    continue
                new_dataset.append((fname,label, cid))
                cluster_centers[label].append(to_torch(features[i]))
            cluster_centers = [torch.stack(cluster_centers[idx]).mean(0) for idx in sorted(cluster_centers.keys())]
            cluster_centers = torch.stack(cluster_centers)

            train_loader = get_train_loader(dataset_target, args.height, args.width, args.batch_size,
                                                   args.workers, args.num_instances, args.iters, trainset=new_dataset)

        cluster_centers = to_torch(normalize(cluster_centers)).to(torch.float32)
        cluster_centers.requires_grad = False
        cluster_centers.detach_()
        num_features = model.classifier.in_features

        # for param in model_ema.parameters():
        #     param.detach_()

        # Optimizer
        lr_ratio = 0.1 ** min(int(epoch / args.step), 2)
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": args.lr*lr_ratio, "weight_decay": args.weight_decay}]
        optimizer = torch.optim.Adam(params)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)
        print('learning rate: ', args.lr*lr_ratio)    

        # Trainer
        # trainer = GCCTrainer(model, model_ema, gat, args.memory_strategy, num_cluster=args.num_clusters, alpha=args.alpha, args=args)
        trainer = GCCTrainer(model, gat, args.memory_strategy, num_cluster=args.num_clusters, alpha=args.alpha, args=args)

        train_loader.new_epoch()
        trainer.train(epoch, train_loader, optimizer, cluster_centers.cuda(),
                    print_freq=args.print_freq, train_iters=len(train_loader),
                    loss_weight=args.loss_weight, beta=args.beta, k=args.k)

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            rank1, mAP = evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)
            is_best = mAP > args.best_map
            if is_best:
                args.best_map = mAP
                args.best_epoch = epoch
            save_model(model, is_best, mAP, epoch, args.logs)
            mAP = str(round(mAP*100, 4))+'%'
            rank1 = str(round(rank1*100))+'%'
            print(f'{{ \'Finished_epoch\': {epoch}, \'mAP\': {mAP}, rank1: {rank1}, \'best\': {round(args.best_map*100, 4)}, \'at_epoch\': {args.best_epoch}, \'this_best\': {1 if is_best else 0}}}')
            print(time_now())
            print('Cost ' + str(datetime.datetime.now() - BEGIN_TIME))
            print(args.logs)
        gc.collect()
        # lr_scheduler.step()

    print('\n' + str(datetime.datetime.now()))
    print('Cost ' + str(datetime.datetime.now() - BEGIN_TIME))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GCMT model")

    # data
    parser.add_argument('-dt', '--dataset_target', type=str, default='market')
    parser.add_argument('--data_dir', type=str, default='E:/dataset/')
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('--num_instances', type=int, default=16)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('-j', '--workers', type=int, default=1)

    # training
    parser.add_argument('--logs', type=str, default='temp')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--loss_weight', type=float, default=0.6)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--step', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=75)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--eps', type=float, default=0.6)
    parser.add_argument('--num_clusters', type=int, default=500)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--init_path', type=str, default='')
    parser.add_argument('--alpha', type=float, default=0.999, help='to update mean teacher')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.00035)
    parser.add_argument('--momentum1', type=float, default=0.9, help='to update memory bank with positive')
    parser.add_argument('--momentum2', type=float, default=0.2, help='to update memory bank with negative')
    parser.add_argument('--best_map', type=float, default=0.)
    parser.add_argument('--debug', action='store_true', default=False)

    parser.add_argument('--memory_strategy', type=str, default='far-update')
    main(parser.parse_args())
