# Reference to https://github.com/lgcollins/FedRep

import copy
import os
import pickle
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import re
import glog as logger

from manifold.manifold_model import ManifoldFlow
from cls_models.mlp import SimpleNet, MLP
from cls_models.cnn import CNNCifar, CNNCifar100, CNNCelebA
from utils.tools import create_filename, model_load
from dataset.transform import ImageClientDataset, ImageTransform

def create_loss_func(nn_loss):
    def loss_func(pred, y, weight=None):
        if type(weight)!=type(None):
            return (nn_loss(pred, y)*weight).mean()
        else:
            return nn_loss(pred, y).mean()
    return loss_func

class FedTrainer(object):
    def __init__(self, args, dataset_clients_train, dataset_clients_test, fed_alg, condition_dim, used_users=None):
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        self.pickle_record = {"test": {}, "train": {}, "clients_train_len":{}, "clients_test_len":{}}
        self.dataset_clients_train = dataset_clients_train
        self.dataset_clients_test = dataset_clients_test
        self.fed_alg = fed_alg
        self.users_list = used_users if used_users else list(self.dataset_clients_train.keys())
        self.condition_dim = condition_dim

        self.generation_alg = ['local_gen', 'fedrep_gen', 'fedavg_gen']
        self.global_alg = ['fedavg', 'fedavg_gen']
        self.local_alg = ['local', 'local_gen']

        self.table_datasets = ['synthetic', 'sinsynthetic', 'adult']

        if 'synthetic' in args.dataset:
            self.cls_model = SimpleNet(input_dim=args.datadim[0], hidden_dim=args.cls_hidden_features, cls_hidden_layer=args.cls_hidden_layer, flatten=True)
        elif 'femnist' in args.dataset:
            self.cls_model = MLP(args.datadim[0], args.num_classes)
        elif args.dataset=='cifar10':
            self.cls_model = CNNCifar(args.num_classes)
        elif args.dataset=='cifar100':
            self.cls_model = CNNCifar100(args.num_classes)
        elif args.dataset=='celeba':
            self.cls_model = CNNCelebA(args.num_classes, drop_out_rate=args.cls_dropout)
        elif args.dataset=='adult':
            self.cls_model = SimpleNet(input_dim=args.datadim[0], hidden_dim=-1, cls_hidden_layer=1, out_dim=2, flatten=False)
        self.cls_model.to(self.device)

        if self.fed_alg in self.generation_alg:
            if 'synthetic' in args.dataset:
                manifold_checkpoint = torch.load(os.path.join(args.manifold_root, args.manifold_pth, 'data/models/manifold.pt'), map_location=self.device)
            else:
                manifold_checkpoint = torch.load(os.path.join(args.manifold_root, args.dataset, 'manifold', args.manifold_pth, 'data/models/manifold.pt'), map_location=self.device)

            latentdim = manifold_checkpoint['inner_transform._transforms.0._transforms.0._transforms.0._permutation'].shape[0]
            args.latentdim = latentdim
            logger.info('manifold latent dim: %d'%latentdim)

            self.manifold_model = ManifoldFlow(self.device, args, data_vector_len=args.manifold_data_vector_len, latent_dim=args.latentdim, pie_epsilon=1.0e-2,
                        condition_dim=condition_dim if args.conditional else None,
                         apply_context_to_outer=(args.outer_condition and args.conditional), clip_pie=False).to(self.device)
            self.manifold_model.load_state_dict(manifold_checkpoint)
            self.manifold_model.eval()

            if not args.conditional:
                assert args.gen_within_closure

        if 'synthetic' in args.dataset:
            self.loss_func = create_loss_func(nn.MSELoss(reduction='none'))
            self.metrics = {'MAE': lambda pred, y: torch.abs(pred-y).mean().item()}
            if args.dataset=='synthetic':
                self.metrics['u_dis'] = lambda x, y, u: torch.abs(torch.matmul(x, u)-y).mean().item()
        elif ('femnist' in args.dataset) or ('cifar' in args.dataset) or (args.dataset=='celeba'):
            self.loss_func = create_loss_func(nn.CrossEntropyLoss(reduction='none'))
            self.metrics = {'ACC': lambda pred, y: (pred.argmax(1)==y).float().mean().item()*100}

    def train(self, args):
        m = max(int(args.user_frac * len(self.users_list)), 1)
        
        lens = {}
        for iii, c in enumerate(self.users_list):
            lens[c] = len(self.dataset_clients_train[c]['x'])
            self.pickle_record["clients_train_len"][c] = len(self.dataset_clients_train[c]['x'])
            self.pickle_record["clients_test_len"][c] = len(self.dataset_clients_test[c]['x'])

        logger.info('clients data lens:'+str(lens))

        for client in self.dataset_clients_train.keys():
            self.dataset_clients_train[client]['data_weight'] = torch.ones([self.dataset_clients_train[client]['x'].shape[0]])
            self.dataset_clients_train[client]['original_data_num'] = self.dataset_clients_train[client]['x'].shape[0]

        if self.fed_alg in self.generation_alg:
            self.dataset_clients_train = self.get_latent_data(args, self.dataset_clients_train)
            if args.share_data:
                self.dataset_clients_train = self.share_latent_data(args, self.dataset_clients_train)

        w_locals = {}
        for user in self.users_list:
            w_local_dict = {}
            for key in self.cls_model.state_dict().keys():
                w_local_dict[key] = self.cls_model.state_dict()[key]
            w_locals[user] = w_local_dict

        if self.fed_alg in self.local_alg:
            w_glob_keys = []
        elif self.fed_alg in self.global_alg:
            w_glob_keys = list(self.cls_model.state_dict().keys())
        elif self.fed_alg in ['fedrep', 'fedrep_gen']:
            w_glob_keys = self.cls_model.weight_keys[:-1]
            w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
        logger.info('w_glob_keys: '+str(w_glob_keys))


        if args.cls_checkpoint:
            self.cls_model = model_load('global', os.path.join(args.cls_checkpoint, 'global.pth'), self.device, cls_model=self.cls_model)
            w_locals = model_load('local', os.path.join(args.cls_checkpoint, 'locals.pth'), self.device, cls_model=None)

            metrics_test, loss_test = self.test_local_all(args, -1, self.cls_model, self.dataset_clients_test, w_locals, w_glob_keys)
            test_metrics_str = ', '.join(['Test %s: %.3f'%(k,v) for k,v in metrics_test.items()])
            logger.info('Loaded model, Test loss: {:.3f}, {:s}'.format(loss_test, test_metrics_str))

        loss_train = []
        for iter in range(args.epochs+1):
            w_glob = {}
            loss_all = 0
            self.pickle_record['train'][iter] = {}

            if iter == args.epochs:
                m = len(self.users_list)

            idxs_users = np.random.choice(self.users_list, m, replace=False)
            total_len=0
            metrics_train_mean = {k: 0.0 for k in self.metrics.keys()}
            for ind, user_n in enumerate(idxs_users):
                net_local = copy.deepcopy(self.cls_model)
                if self.fed_alg not in self.global_alg:
                    w_local = net_local.state_dict()
                    for k in w_locals[user_n].keys():
                        if k not in w_glob_keys:
                            w_local[k] = w_locals[user_n][k]
                    net_local.load_state_dict(w_local)

                w_local, loss, metrics_train_local = self.local_train(args, net=net_local.to(self.device), 
                                                            original_dataset_train=self.dataset_clients_train[user_n],
                                                             w_glob_keys=w_glob_keys, lr=args.lr, client_name=user_n)

                loss_all += loss*lens[user_n]

                self.pickle_record["train"][iter][user_n] = {'loss':loss}
                for k in metrics_train_local.keys():
                    metrics_train_mean[k] += metrics_train_local[k]*lens[user_n]
                    self.pickle_record["train"][iter][user_n][k] = metrics_train_local[k]

                total_len += lens[user_n]
                if len(w_glob) == 0:
                    w_glob = copy.deepcopy(w_local)
                    for k,key in enumerate(self.cls_model.state_dict().keys()):
                        w_glob[key] = w_local[key]*lens[user_n]
                        w_locals[user_n][key] = w_local[key]
                else:
                    for k,key in enumerate(self.cls_model.state_dict().keys()):
                        w_glob[key] += w_local[key]*lens[user_n]
                        w_locals[user_n][key] = w_local[key]
            
            loss_avg = loss_all/total_len
            loss_train.append(loss_avg)
            for k in metrics_train_local.keys():
                metrics_train_mean[k] = metrics_train_mean[k]/total_len

            # get weighted average for global weights
            for k in self.cls_model.state_dict().keys():
                w_glob[k] = torch.div(w_glob[k], total_len)

            if args.epochs != iter:
                self.cls_model.load_state_dict(w_glob)

            if iter % args.test_freq==args.test_freq-1 or iter>=args.epochs-10:
                metrics_test, loss_test = self.test_local_all(args, iter, self.cls_model, self.dataset_clients_test, w_locals, w_glob_keys)

                train_metrics_str = ', '.join(['Train %s: %.3f'%(k,metrics_train_mean[k]) for k in metrics_train_local.keys()])
                test_metrics_str = ', '.join(['Test %s: %.3f'%(k,v) for k,v in metrics_test.items()])
                logger.info('Round {:3d}, Train loss: {:.3f}, {:s}; Test loss: {:.3f}, {:s}'.format(iter, loss_avg, train_metrics_str, loss_test, test_metrics_str))

                with open(create_filename('pickle', None, args), "wb") as f:
                    pickle.dump(self.pickle_record, f)

            if (iter % args.save_every==args.save_every-1) or iter>=args.epochs-10:
                torch.save(self.cls_model.state_dict(), create_filename('cls_model', 'global', args))
                torch.save(w_locals, create_filename('cls_model', 'locals', args))

        with open(create_filename('pickle', None, args), "wb") as f:
            pickle.dump(self.pickle_record, f)

    def add_value_to_pickle(self, mode, epoch, k, v):
        if epoch not in self.pickle_record[mode].keys():
            self.pickle_record[mode][epoch] = {}

        for user in self.users_list:
            if user not in self.pickle_record[mode][epoch].keys():
                self.pickle_record[mode][epoch][user] = {k:v}
            else:
                self.pickle_record[mode][epoch][user][k]=v

    def get_latent_data(self, args, dataset_clients):
        logger.info('Start latent vector extraction with manifold')
        self.manifold_model.eval()
        for ind, user_n in enumerate(self.users_list):

            ldr = DataLoader(TensorDataset(dataset_clients[user_n]['x'], dataset_clients[user_n]['y']),
                                         batch_size=args.manifold_batchsize, shuffle=False)
            latent_vectors = []
            for batch_idx, (x, y) in enumerate(ldr):
                x, y = x.to(self.device), y.to(self.device)
                context = None
                if (not args.outer_image) and len(x.shape)>2:
                    x = x.reshape(x.shape[0], -1)
                if args.conditional:
                    context = F.one_hot(y, num_classes=self.condition_dim).to(x.dtype)

                if (args.dataset in self.table_datasets) and (not args.conditional):
                    x = torch.cat([x, y.float().unsqueeze(1)], dim=1)

                with torch.no_grad():
                    u, h_manifold, h_orthogonal, _, _ = self.manifold_model._encode(x, context=context)
                latent_vector = u if args.latent_in_u else h_manifold
                latent_vectors.append(latent_vector.detach())

            latent_vectors_all = torch.cat(latent_vectors, dim=0)
            dataset_clients[user_n]['latent_vectors'] = latent_vectors_all

            if (not args.conditional) and (not args.gen_share_each_label):
                dataset_clients[user_n]['latent_min_labels'] = latent_vectors_all.min(0)[0]
                dataset_clients[user_n]['latent_max_labels'] = latent_vectors_all.max(0)[0]
            else:
                dataset_clients[user_n]['latent_min_labels'] = {}
                dataset_clients[user_n]['latent_max_labels'] = {}
                labels_set = set(dataset_clients[user_n]['y'].numpy().tolist())
                for label in labels_set:
                    latent_vectors_label = latent_vectors_all[dataset_clients[user_n]['y']==label]
                    dataset_clients[user_n]['latent_min_labels'][label] = latent_vectors_label.min(0)[0]
                    dataset_clients[user_n]['latent_max_labels'][label] = latent_vectors_label.max(0)[0]

        logger.info('Finish latent vector extraction with manifold')
        return dataset_clients

    def judge_in_closure(self, latent_min_rate, latent_max_rate, latent_vectors):
        return ((latent_vectors<=latent_max_rate)*(latent_vectors>=latent_min_rate)).sum(1)==latent_vectors.shape[1]

    def share_latent_data(self, args, original_dataset_clients):
        dataset_clients = copy.deepcopy(original_dataset_clients)
        original_data_num_all = 0
        total_share = 0
        correct = 0
        for ind_i in range(len(self.users_list)):
            latent_min = original_dataset_clients[self.users_list[ind_i]]['latent_min_labels']
            latent_max = original_dataset_clients[self.users_list[ind_i]]['latent_max_labels']
            if args.gen_share_each_label:
                latent_min_rate, latent_max_rate = dict(), dict()
                for k in latent_min.keys():
                    adjust_len = (latent_max[k]-latent_min[k])*(1-args.closure_rate)/2
                    latent_min_rate[k] = latent_min[k]+adjust_len
                    latent_max_rate[k] = latent_max[k]-adjust_len
            else:
                adjust_len = (latent_max-latent_min)*(1-args.closure_rate)/2
                latent_min_rate = latent_min+adjust_len
                latent_max_rate = latent_max-adjust_len

            labels_i_set = set(original_dataset_clients[self.users_list[ind_i]]['y'].numpy().tolist())
            original_data_num = len(original_dataset_clients[self.users_list[ind_i]]['x'])
            original_data_num_all += original_data_num

            share_data_x_list = []
            share_data_y_list = []
            share_data_ind = []

            for ind_j in range(len(self.users_list)):
                if ind_i==ind_j:
                    continue
                latent_vectors_j = original_dataset_clients[self.users_list[ind_j]]['latent_vectors']

                if args.gen_share_each_label:
                    in_convex_hull = torch.zeros([latent_vectors_j.shape[0]], device=latent_vectors_j.device, dtype=torch.bool)
                    for label_i in labels_i_set:
                        label_i_inds = torch.where(original_dataset_clients[self.users_list[ind_j]]['y']==label_i)[0]
                        in_convex_hull_y = self.judge_in_closure(latent_min_rate[label_i], latent_max_rate[label_i],
                                                         latent_vectors_j[label_i_inds])
                        in_convex_hull[label_i_inds] = in_convex_hull_y
                else:
                    in_convex_hull = self.judge_in_closure(latent_min_rate, latent_max_rate, latent_vectors_j)

                share_data_x_list.append(original_dataset_clients[self.users_list[ind_j]]['x'][in_convex_hull])
                share_data_y_list.append(original_dataset_clients[self.users_list[ind_j]]['y'][in_convex_hull])
                share_data_ind+=[(self.users_list[ind_j], data_ind_j.item()) for data_ind_j in torch.where(in_convex_hull)[0]]

                total_share+=in_convex_hull.sum().item()
                if args.dataset=='synthetic':
                    if (self.users_list[ind_i]-2.5)*(self.users_list[ind_j]-2.5)>0:
                        correct+=in_convex_hull.sum().item()
                elif 'cifar' in args.dataset or 'femnist' in args.dataset:
                    correct_y = list(filter(lambda x:x.item() in labels_i_set, original_dataset_clients[self.users_list[ind_j]]['y'][in_convex_hull]))
                    correct += len(correct_y)
                elif args.dataset=='sinsynthetic':
                    x_j = original_dataset_clients[self.users_list[ind_j]]['x'][:, 0]
                    x_ranges = original_dataset_clients[self.users_list[ind_i]]['u_m']
                    correct_flag = torch.zeros([latent_vectors_j.shape[0]], device=x_j.device, dtype=torch.bool)
                    if not ((self.users_list[ind_i]-args.num_users//2+0.5)*(self.users_list[ind_j]-args.num_users//2+0.5)<0):    
                        for x_range in x_ranges:
                            correct_flag_i = (x_j>=x_range[0])*(x_j<=x_range[1])
                            correct_flag += correct_flag_i
                        correct_flag = correct_flag>0

                    correct += (correct_flag.to(in_convex_hull.device)*in_convex_hull).sum().item()

            dataset_clients[self.users_list[ind_i]]['share_data_ind'] = share_data_ind
            if len(share_data_x_list)>0:
                dataset_clients[self.users_list[ind_i]]['x'] = torch.cat([original_dataset_clients[self.users_list[ind_i]]['x'],
                                                                        torch.cat(share_data_x_list, dim=0)], dim=0)
                dataset_clients[self.users_list[ind_i]]['y'] = torch.cat([original_dataset_clients[self.users_list[ind_i]]['y'],
                                                                        torch.cat(share_data_y_list, dim=0)], dim=0)

            dataset_clients[self.users_list[ind_i]]['data_weight'] = torch.ones([dataset_clients[self.users_list[ind_i]]['x'].shape[0]])
            dataset_clients[self.users_list[ind_i]]['data_weight'][original_data_num:] = args.share_data_weight
        
        if ('synthetic' in args.dataset) or ('cifar' in args.dataset) or ('femnist' in args.dataset):
            share_data_rate = float(total_share)/original_data_num_all
            share_data_correct_rate = float(correct)/total_share*100.0 if total_share>0 else 0
            logger.info('share data rate: %.2f, share data correct rate: %.2f'%(share_data_rate, share_data_correct_rate))
            self.add_value_to_pickle('test', -2, 'share_data_rate', share_data_rate)
            self.add_value_to_pickle('test', -2, 'share_data_correct_rate', share_data_correct_rate)

        if total_share==0:
            raise KeyboardInterrupt('total_share=0')

        return dataset_clients

    def augment_training_data(self, args, dataset_train):
        original_data_num = dataset_train['original_data_num']

        if (self.fed_alg in self.generation_alg):
            generate_data_num = int(original_data_num*args.generate_data_ratio)
            if generate_data_num==0:
                return dataset_train, original_data_num
            labels = set(dataset_train['y'].cpu().numpy().tolist())
            data_num_eachclass = generate_data_num//len(labels)
            data_num_eachclass = max(data_num_eachclass, 1)

            if not args.gen_share_each_label:
                latent_min = dataset_train['latent_min_labels'].unsqueeze(0)
                latent_max = dataset_train['latent_max_labels'].unsqueeze(0)
                assert args.gen_within_closure
                generated_x_all = self.generate_x_from_latent(generate_data_num, args, latent_min, latent_max, context=None)
                if args.dataset in self.table_datasets:
                    if 'synthetic' in args.dataset:
                        y_all = generated_x_all[:, -1]
                    else:
                        y_all = torch.round(torch.clamp(generated_x_all[:, -1], min=0, max=1)).long()
                else:
                    raise KeyError()

                generated_x_all = generated_x_all[:, :-1]
                generated_num = generate_data_num

            else:
                generated_x_all = []
                y_all = []
                for label in labels:
                    context = None
                    if args.conditional:
                        context = F.one_hot(torch.Tensor([label]).long(), num_classes=self.condition_dim).to(self.device, torch.float) # [1,10]

                    latent_min = dataset_train['latent_min_labels'][label].unsqueeze(0)
                    latent_max = dataset_train['latent_max_labels'][label].unsqueeze(0)
                    generated_x = self.generate_x_from_latent(data_num_eachclass, args, latent_min, latent_max, context)
                    if (args.dataset in self.table_datasets) and (not args.conditional):
                        generated_x = generated_x[:, :-1]
                    generated_x_all.append(generated_x)
                    y_all += ([label]*generated_x.shape[0])
                
                y_all = torch.Tensor(y_all).long().to(dataset_train['x'].device)
                generated_x_all = torch.cat(generated_x_all, dim=0)
                generated_num = data_num_eachclass*len(labels)

            generated_x_all = generated_x_all.detach().to(dataset_train['x'].device)
            if not ('synthetic' in args.dataset):
                generated_x_all = torch.clamp(generated_x_all, min=0, max=1)
            else:
                generated_x_all[:, args.data_var_dim:] = 0.0

            if len(generated_x_all.shape)!=len(args.datadim)+1:
                generated_x_all = generated_x_all.reshape(generated_x_all.shape[0], *args.datadim)
            dataset_train['x'] = torch.cat([dataset_train['x'], generated_x_all], dim=0)
            dataset_train['y'] = torch.cat([dataset_train['y'], y_all], dim=0)

                
            dataset_train['data_weight'] = torch.cat([dataset_train['data_weight'], torch.Tensor([args.generate_data_weight]*generated_num)], dim=0)

        return dataset_train, original_data_num

    def generate_x_from_latent(self, generate_num, args, latent_min, latent_max, context):
        if args.gen_within_closure:
            if args.generate_data_prior=='uniform':
                generated_vector = torch.rand(generate_num, latent_min.shape[1], device=latent_max.device)*(latent_max-latent_min)+latent_min
            elif args.generate_data_prior=='normal':
                generated_vector = torch.randn(generate_num, latent_min.shape[1], device=latent_max.device)*(latent_max-latent_min)/4+(latent_min+latent_max)/2
            context_vector = context.repeat(generated_vector.shape[0], 1) if args.conditional else None
            with torch.no_grad():
                if args.latent_in_u:
                    generated_x, _, _, _, _ = self.manifold_model._decode(generated_vector.to(self.device), mode='projection', context=context_vector)
                else:
                    generated_x, _, _, _ = self.manifold_model._decode_h(generated_vector.to(self.device),
                                            mode='projection', h_orthogonal=None, context=context_vector)
        else:
            with torch.no_grad():
                generated_x = self.manifold_model.sample(n=generate_num, context=context.repeat(generate_num, 1) if args.conditional else None)
        
        generated_x = self.eliminate_nan_data(generated_x.detach())
        return generated_x

    def eliminate_nan_data(self, generate_x):
        nan_num = torch.isnan(generate_x).sum([i for i in range(1, generate_x.dim())])
        if torch.any(nan_num>0):
            logger.info('%d nan data dropped'%((nan_num>0).sum().item()))
            return generate_x[nan_num==0]
        return generate_x

    def set_require_grad(self, args, net, w_glob_keys, local_iter):
        if (local_iter < args.head_epochs and (self.fed_alg in ['fedrep', 'fedrep_gen'])):
            for name, param in net.named_parameters():
                if name in w_glob_keys:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        
        # then do local epochs for the representation
        elif local_iter == args.head_epochs and (self.fed_alg in ['fedrep', 'fedrep_gen']):
            for name, param in net.named_parameters():
                if name in w_glob_keys:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        elif self.fed_alg not in ['fedrep', 'fedrep_gen']:
            for name, param in net.named_parameters():
                param.requires_grad = True 

        return net

    def local_train(self, args, net, original_dataset_train, w_glob_keys, lr, client_name):
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
            [     
                {'params': weight_p, 'weight_decay': args.weightdecay},
                {'params': bias_p, 'weight_decay':0}
            ],
            lr=lr, momentum=0.5
        )

        original_dataset_train['x'] = original_dataset_train['x'].to(self.device)
        original_dataset_train['y'] = original_dataset_train['y'].to(self.device)

        dataset_train, original_data_num = self.augment_training_data(args, copy.deepcopy(original_dataset_train))

        ldr_train = DataLoader(self.get_client_dataset(args, dataset_train['x'], dataset_train['y'], dataset_train['data_weight'], True),
                                 batch_size=args.batchsize, shuffle=True)

        net.train()
        epoch_loss = []
        metrics_train_local = {k: 0 for k in self.metrics.keys() if k!='AUC'}
        tot = 0
        for iter in range(args.local_epochs):
            net = self.set_require_grad(args, net, w_glob_keys, iter)

            batch_loss = []
            for batch_idx, (x, y, weight) in enumerate(ldr_train):
                x, y, weight = x.to(self.device), y.to(self.device), weight.to(self.device)
                net.zero_grad()
                probs = net(x)
                loss = self.loss_func(probs, y, weight)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

                if torch.isnan(loss).any():
                    raise Exception('loss nan')

                for k in metrics_train_local.keys():
                    if k=='u_dis' and args.dataset=='synthetic':
                        metrics_train_local[k] += self.metrics[k](x=dataset_train['x'][original_data_num:], 
                                                    y=dataset_train['y'][original_data_num:], u=dataset_train['u_m'])*(x.shape[0]-original_data_num)
                    else:
                        metrics_train_local[k] += self.metrics[k](probs, y)*x.shape[0]
                tot += x.shape[0]
                
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            if args.gen_every_localepoch and (self.fed_alg in self.generation_alg):
                dataset_train, original_data_num = self.augment_training_data(args, copy.deepcopy(original_dataset_train))
                ldr_train = DataLoader(self.get_client_dataset(args, dataset_train['x'], dataset_train['y'], dataset_train['data_weight'], True),
                                         batch_size=args.batchsize, shuffle=True)

        for k in metrics_train_local.keys():
            metrics_train_local[k] = metrics_train_local[k]/tot

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), metrics_train_local
    
    def get_client_dataset(self, args, x, y, weight, train_flag):
        if 'cifar' in args.dataset:
            transform = ImageTransform(32, 4, rand_noise=False, train=train_flag, flatten=(not args.outer_image), dataset=args.dataset,
                         normalize=True, from_tensor=True)
            dataset = ImageClientDataset(x, y, weight, transform)
        else:
            tensor_list = [x, y]
            if type(weight)!=type(None):
                tensor_list.append(weight)
            dataset = TensorDataset(*tensor_list)
        return dataset

    def test_local_all(self, args, epoch, net, dataset_clients_test, w_locals, w_glob_keys=None):
        tot = 0
        num_idxxs = len(self.users_list)
        metrics_test_local = {k: np.zeros(num_idxxs) for k in self.metrics.keys() if k!='u_dis'}
        loss_test_local = np.zeros(num_idxxs)
        self.pickle_record['test'][epoch] = {}

        for ind, user_n in enumerate(self.users_list):
            net_local = copy.deepcopy(net)
            if self.fed_alg not in self.global_alg:
                w_local = net_local.state_dict()
                for k in w_locals[user_n].keys():
                    if w_glob_keys is not None and k not in w_glob_keys:
                        w_local[k] = w_locals[user_n][k]
                net_local.load_state_dict(w_local)
            net_local.eval()

            ldr_test = DataLoader(self.get_client_dataset(args, dataset_clients_test[user_n]['x'], dataset_clients_test[user_n]['y'], weight=None, train_flag=False),
                                         batch_size=args.batchsize, shuffle=False)

            preds_all = []
            y_all = []
            for batch_idx, (x, y) in enumerate(ldr_test):
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    preds = net_local(x)
                    loss = self.loss_func(preds, y)
                    preds_all.append(preds.detach())
                    y_all.append(y)

                loss_test_local[ind] += loss.item()*x.shape[0]

            for k in metrics_test_local.keys():
                metrics_test_local[k][ind] += self.metrics[k](torch.cat(preds_all), torch.cat(y_all))*len(dataset_clients_test[user_n]['x'])

            tot += len(dataset_clients_test[user_n]['x'])

            self.pickle_record['test'][epoch][user_n] = {'loss':loss_test_local[ind]/len(dataset_clients_test[user_n]['x'])}
            for k in metrics_test_local.keys():
                self.pickle_record['test'][epoch][user_n][k] = metrics_test_local[k][ind]/len(dataset_clients_test[user_n]['x'])

        for k in metrics_test_local.keys():
            metrics_test_local[k] = metrics_test_local[k].sum()/tot
        return  metrics_test_local, sum(loss_test_local)/tot
