from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
from config import parser
from models.base_models import NCModel, LPModel, RENCModel, RELPModel#, #ADVNCModel, ADVLPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
    from torch_geometric.utils import train_test_split_edges

    dataset = Planetoid("data/", args.dataset, transform=T.NormalizeFeatures())
    data_pyg = dataset[0]
    all_edge_index = data_pyg.edge_index
    data_pyg = train_test_split_edges(data_pyg, 0.05, 0.1)

    reserve_mark = 0

    if args.task == 'nc':
        reserve_mark = 0
    else:
        args.task = 'nc'
        reserve_mark = 1
    # Load data
    data = load_data(args, os.path.join('data/', args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape
    if args.task == 'nc':
        # Model = ADVNCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            print(' ')
            # Model = ADVLPModel
        else:
            Model = RECModel
            # No validation for reconstruction task
            args.eval_freq = args.epochs + 1

    #transfer loading
    if reserve_mark == 1:
        args.task = 'lp'
        # reset reserve mark
        reserve_mark = 0

    if args.task == 'lp':
        reserve_mark = 0
    else:
        args.task = 'lp'
        reserve_mark = 1

    data1 = load_data(args, os.path.join('data/', args.dataset))
    args.n_nodes, args.feat_dim = data1['features'].shape
    if args.task == 'nc':
        # Model = ADVNCModel
        args.n_classes = int(data1['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        print('*****')
        args.nb_false_edges = len(data1['train_edges_false'])
        args.nb_edges = len(data1['train_edges'])
        if args.task == 'lp':
            print(' ')
            # Model = ADVLPModel
        else:
            Model = RECModel
            # No validation for reconstruction task
            args.eval_freq = args.epochs + 1

    if reserve_mark == 1:
        args.task = 'nc'

    # if args.task == 'nc':
    #     Model = ADVNCModel
    # else:
    #     Model = ADVLPModel

    print(data_pyg.x)
    print(data['features'])
    print((data_pyg.x == data['features']).all())





    # if not args.lr_reduce_freq:
    #     args.lr_reduce_freq = args.epochs
    #
    # # Model and optimizer
    # model = Model(args)
    # logging.info(str(model))
    # optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
    #                                                 weight_decay=args.weight_decay)
    # optimizer_en = getattr(optimizers, args.optimizer)(params=model.encoder.parameters(), lr=args.lr,
    #                                                    weight_decay=args.weight_decay)
    # optimizer_de = getattr(optimizers, args.optimizer)(params=model.net.parameters(), lr=args.lr,
    #                                                 weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=int(args.lr_reduce_freq),
    #     gamma=float(args.gamma)
    # )
    # lr_scheduler_en = torch.optim.lr_scheduler.StepLR(
    #     optimizer_en,
    #     step_size=int(args.lr_reduce_freq),
    #     gamma=float(args.gamma)
    # )
    # lr_scheduler_de = torch.optim.lr_scheduler.StepLR(
    #     optimizer_de,
    #     step_size=int(args.lr_reduce_freq),
    #     gamma=float(args.gamma)
    # )
    # tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    # logging.info(f"Total number of parameters: {tot_params}")
    # if args.cuda is not None and int(args.cuda) >= 0 :
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    #     model = model.to(args.device)
    #     for x, val in data.items():
    #         if torch.is_tensor(data[x]):
    #             data[x] = data[x].to(args.device)
    #     for x, val in data1.items():
    #         if torch.is_tensor(data1[x]):
    #             data1[x] = data1[x].to(args.device)
    # # Train model
    # t_total = time.time()
    # counter = 0
    # best_val_metrics = model.init_metric_dict()
    # best_test_metrics = None
    # best_emb = None
    # for epoch in range(args.epochs):
    #     t = time.time()
    #     model.train()
    #     # if epoch%3==0:
    #     optimizer.zero_grad()
    #     optimizer_de.zero_grad()
    #     embeddings = model.encode(data['features'], data['adj_train_norm'])
    #     train_metrics = model.compute_metrics(embeddings, data, 'train')
    #     train_metrics['loss'].backward()
    #     if args.grad_clip is not None:
    #         max_norm = float(args.grad_clip)
    #         all_params = list(model.parameters())
    #         for param in all_params:
    #             torch.nn.utils.clip_grad_norm_(param, max_norm)
    #     optimizer.step()
    #     # model.save_net()
    #     # model.save_emb()
    #     # lr_scheduler.step()
    #
    # # if epoch%3==1:
    # #     if epoch > 100:
    # #     if (args.model == 'GCN' or args.model == 'GAT' and epoch > 100) or (args.model == 'HGCN' and epoch > 1000):
    #     optimizer_en.zero_grad()
    #     # model.load_emb()
    #     embeddings1 = model.encode(data1['features'], data1['adj_train_norm'])
    #     train_metrics1 = model.compute_metrics1(embeddings1, data1, 'train')
    #     loss1 = -(train_metrics1['loss'] - train_metrics1['loss_shuffle'])
    #     loss1.backward()
    #     optimizer_en.step()
    #     # model.load_net()
    # #     # lr_scheduler.step()
    # #
    # # # if epoch%3==2:
    #     optimizer_de.zero_grad()
    #     embeddings2 = model.encode(data1['features'], data1['adj_train_norm']).detach_()
    #     train_metrics2 = model.compute_metrics1(embeddings2, data1, 'train')
    #     loss2 = (train_metrics2['loss'] - train_metrics2['loss_shuffle'])
    #     loss2.backward()
    #     optimizer_de.step()
    #     lr_scheduler.step()
    #     lr_scheduler_de.step()
    #     lr_scheduler_en.step()
    #     # if epoch<100:
    #     #     train_metrics2 = train_metrics
    #     if (epoch + 1) % args.log_freq == 0:
    #         logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
    #                                'lr: {}'.format(lr_scheduler.get_lr()[0]),
    #                                format_metrics(train_metrics, 'train'),
    #                                format_metrics(train_metrics2, 'train'),
    #                                'time: {:.4f}s'.format(time.time() - t)
    #                                ]))
    #         if not best_val_metrics == None:
    #             logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    #             logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics1, 'val')]))
    #             logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))
    #             logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics1, 'test')]))
    #     if (epoch + 1) % args.eval_freq == 0:
    #         model.eval()
    #         embeddings = model.encode(data['features'], data['adj_train_norm'])
    #         val_metrics = model.compute_metrics(embeddings, data, 'val')
    #         embeddings1 = model.encode(data1['features'], data1['adj_train_norm'])
    #         val_metrics1 = model.compute_metrics1(embeddings1, data1, 'val')
    #         if (epoch + 1) % args.log_freq == 0:
    #             logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val'), format_metrics(val_metrics1, 'val')]))
    #
    #         embeddings = model.encode(data['features'], data['adj_train_norm'])
    #         test_metrics = model.compute_metrics(embeddings, data, 'test')
    #         embeddings1 = model.encode(data1['features'], data1['adj_train_norm'])
    #         test_metrics1 = model.compute_metrics1(embeddings1, data1, 'test')
    #         if (epoch + 1) % args.log_freq == 0:
    #             logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(test_metrics, 'test'),
    #                                    format_metrics(test_metrics1, 'test')]))
    #         if model.has_improved(best_val_metrics, val_metrics):
    #             best_test_metrics = model.compute_metrics(embeddings, data, 'test')
    #             best_test_metrics1 = model.compute_metrics1(embeddings1, data1, 'test')
    #             best_emb = embeddings.cpu()
    #             if args.save:
    #                 np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
    #             best_val_metrics = val_metrics
    #             best_val_metrics1 = val_metrics1
    #             counter = 0
    #         # else:
    #         #     counter += 1
    #         #     if counter == args.patience and epoch > args.min_epochs:
    #         #         logging.info("Early stopping")
    #         #         break
    #
    # logging.info("Optimization Finished!")
    # logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # if not best_test_metrics:
    #     model.eval()
    #     best_emb = model.encode(data['features'], data['adj_train_norm'])
    #     best_test_metrics = model.compute_metrics(best_emb, data, 'test')
    # logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    # logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))
    # if args.save:
    #     np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
    #     if hasattr(model.encoder, 'att_adj'):
    #         filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
    #         pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
    #         print('Dumped attention adj: ' + filename)
    #
    #     json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
    #     torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
    #     logging.info(f"Saved model in {save_dir}")

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
