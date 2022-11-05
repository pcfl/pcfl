import json
import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
import os
import numpy as np
import random

from manifold.various import product
from dataset.json_data import All_data_json, read_data, Adult_NF_dataset
from dataset.transform import ImageTransform, Onehot_transform
from dataset.synthetic_data import generate_synthetic_data, generate_sinsynthetic_data

def get_manifold_dataset(args):
    if 'synthetic' in args.dataset:
        data_train, data_test, dict_users_train, dict_users_test, u_m_dict = generate_sinsynthetic_data(args)
        assert data_train.shape[0]==args.trainN*args.num_users, 'data loaded, the length is not the same with args.trainN*args.num_users'
        dataset_train = TensorDataset(torch.Tensor(data_train))
        dataset_test = TensorDataset(torch.Tensor(data_test))

        condition_dim = None
        img_size = None
        data_dim = args.data_var_dim+args.data_cons_dim+1
        latent_dim = args.data_var_dim+1
        is_image = False

    elif 'femnist' in args.dataset:
        data_pth = os.path.join(args.data_pth, 'leaf-master/data', args.dataset, 'data')
        dataset_train = All_data_json(data_pth, train_flag=True)
        dataset_test = All_data_json(data_pth, train_flag=False)
        
        condition_dim = 10 if args.conditional else None
        img_size = [1,28,28]
        latent_dim = args.latentdim
        assert latent_dim>0
        is_image = True
    
    elif args.dataset=='cifar10':
        num_class = 10
        dataset_train = datasets.CIFAR10(os.path.join(args.data_pth, args.dataset), train=True, download=True, 
                                        transform=ImageTransform(32, 4, True, True, flatten=(not args.outer_image)),
                                        target_transform=Onehot_transform(num_class=num_class)) # 50000
        dataset_test = datasets.CIFAR10(os.path.join(args.data_pth, args.dataset), train=False, download=True,
                                        transform=ImageTransform(32, 4, True, False, flatten=(not args.outer_image)), 
                                        target_transform=Onehot_transform(num_class=num_class)) # 10000

        condition_dim = num_class if args.conditional else None
        img_size = [3,32,32]
        latent_dim = args.latentdim
        assert latent_dim>0
        is_image = True
    
    elif args.dataset=='cifar100':
        num_class = 100
        dataset_train = datasets.CIFAR10(os.path.join(args.data_pth, args.dataset), train=True, download=True, 
                                        transform=ImageTransform(32, 4, True, True, flatten=(not args.outer_image)),
                                        target_transform=Onehot_transform(num_class=num_class)) # 50000
        dataset_test = datasets.CIFAR10(os.path.join(args.data_pth, args.dataset), train=False, download=True,
                                        transform=ImageTransform(32, 4, True, False, flatten=(not args.outer_image)), 
                                        target_transform=Onehot_transform(num_class=num_class)) # 10000

        condition_dim = num_class if args.conditional else None
        img_size = [3,32,32]
        latent_dim = args.latentdim
        assert latent_dim>0
        is_image = True

    elif args.dataset=='celeba':
        num_class = 2
        data_pth = os.path.join(args.data_pth, args.dataset)
        dataset_train = All_data_json(data_pth, num_class, train_flag=True, feature_transform=lambda x:np.transpose(x, [0, 3, 1, 2])/255.0)
        dataset_test = All_data_json(data_pth, num_class, train_flag=False, feature_transform=lambda x:np.transpose(x, [0, 3, 1, 2])/255.0)

        condition_dim = num_class if args.conditional else None
        img_size = [3,64,64]
        latent_dim = args.latentdim
        assert latent_dim>0
        is_image = True

    elif args.dataset=='adult':
        num_class = 2
        train_data_dir = os.path.join(args.data_pth, "adult/train/mytrain.json")
        test_data_dir = os.path.join(args.data_pth, "adult/test/mytest.json")
        dataset_train = Adult_NF_dataset(train_data_dir, num_class, train_flag=True, concat_label=(not args.conditional))
        dataset_test = Adult_NF_dataset(test_data_dir, num_class, train_flag=False, concat_label=(not args.conditional))

        data_dim = dataset_train.features.shape[1]
        condition_dim = num_class if args.conditional else None
        img_size = None
        latent_dim = args.latentdim
        assert latent_dim>0
        is_image = False

    if 'femnist' in args.dataset:
        f_label_toname = lambda num:chr(num+36+61)
    elif 'cifar' in args.dataset:
        f_label_toname = lambda num:dataset_train.classes[num]
    elif args.dataset=='celeba':
        f_label_toname = lambda num:['Not smile', 'Smiling'][num]
    else:
        f_label_toname = None
        assert not is_image

    if img_size:
        data_dim = img_size[0]*img_size[1]*img_size[1]

    return dataset_train, dataset_test, condition_dim, img_size, data_dim, latent_dim, is_image, f_label_toname



def get_cls_dataset(args):
    if 'synthetic' in args.dataset:
        data_train, data_test, dict_users_train, dict_users_test, u_m_dict = generate_sinsynthetic_data(args)
        assert data_train.shape[0]==args.trainN*args.num_users, 'data loaded, the length is not the same with args.trainN'

        data_dim = [args.data_var_dim+args.data_cons_dim]
        latent_dim = args.data_var_dim+1
        assert data_train.shape[1]==data_dim[0]+1

        def distribute2clients(data, dict_users):
            dataset_clients = {}
            for client in dict_users.keys():
                data_i = data[dict_users[client]]
                dataset_clients[client] = {'x': torch.Tensor(data_i[:, :data_dim[0]]), 'y': torch.Tensor(data_i[:, data_dim[0]])}
                dataset_clients[client]['u_m'] = torch.Tensor(u_m_dict[client])
            return dataset_clients

        dataset_clients_train = distribute2clients(data_train, dict_users_train)
        dataset_clients_test = distribute2clients(data_test, dict_users_test)
        condition_dim = None
        num_classes = None
        img_size = None
        manifold_data_vector_len = data_dim[0]+1
    
    elif 'femnist' in args.dataset:
        num_classes = 10
        data_pth = os.path.join(args.data_pth, 'leaf-master/data', args.dataset, 'data')
        clients, _, train_data, test_data = read_data(os.path.join(data_pth, 'train'), os.path.join(data_pth, 'test'))
        dataset_clients_train = {user: {'x':torch.Tensor(train_data[user]['x']), 'y':torch.Tensor(train_data[user]['y']).long()} for user in train_data.keys()}
        dataset_clients_test = {user: {'x':torch.Tensor(test_data[user]['x']), 'y':torch.Tensor(test_data[user]['y']).long()} for user in test_data.keys()}
        condition_dim = num_classes if args.conditional else None
        data_dim = [784]
        latent_dim = args.latentdim
        img_size = None
        manifold_data_vector_len = data_dim[0]

    elif args.dataset=='cifar10':
        num_classes = 10
        dataset_train = datasets.CIFAR10(os.path.join(args.data_pth, args.dataset), train=True, download=True, 
                                        transform=None, target_transform=None) # 50000
        dataset_test = datasets.CIFAR10(os.path.join(args.data_pth, args.dataset), train=False, download=True,
                                        transform=None, target_transform=None) # 10000

        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, num_classes, rand_set_all=rand_set_all)

        dataset_clients_train = user_idx_to_dataset(dict_users_train, dataset_train, fx=transforms.ToTensor())
        dataset_clients_test = user_idx_to_dataset(dict_users_test, dataset_test, fx=transforms.ToTensor())

        condition_dim = num_classes if args.conditional else None
        data_dim = [3,32,32]
        latent_dim = args.latentdim
        img_size = data_dim
        mflow_data_vector_len = product(data_dim)

    elif args.dataset=='cifar100':
        num_classes = 100
        dataset_train = datasets.CIFAR100(os.path.join(args.data_pth, args.dataset), train=True, download=True, 
                                        transform=ImageTransform(32, 4, rand_noise=False, train=True, flatten=(not args.outer_image), dataset='cifar100', normalize=True),
                                        target_transform=None) # 50000
        dataset_test = datasets.CIFAR100(os.path.join(args.data_pth, args.dataset), train=False, download=True,
                                        transform=ImageTransform(32, 4, rand_noise=False, train=True, flatten=(not args.outer_image), dataset='cifar100', normalize=True), 
                                        target_transform=None) # 10000

        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, num_classes, rand_set_all=rand_set_all)

        dataset_clients_train = user_idx_to_dataset(dict_users_train, dataset_train, fx=transforms.ToTensor())
        dataset_clients_test = user_idx_to_dataset(dict_users_test, dataset_test, fx=transforms.ToTensor())

        condition_dim = num_classes if args.conditional else None
        data_dim = [3,32,32]
        latent_dim = args.latentdim
        img_size = data_dim
        mflow_data_vector_len = product(data_dim)
    
    elif 'celeba' in args.dataset:
        num_classes = 2
        data_pth = os.path.join(args.data_pth, args.dataset)
        clients, _, train_data, test_data = read_data(os.path.join(data_pth, 'train'), os.path.join(data_pth, 'test'))
        transform_x = lambda x:np.transpose(x, [0, 3, 1, 2])/255.0
        dataset_clients_train = {user: {'x':torch.Tensor(transform_x(train_data[user]['x'])), 'y':torch.Tensor(train_data[user]['y']).long()} for user in train_data.keys()}
        dataset_clients_test = {user: {'x':torch.Tensor(transform_x(test_data[user]['x'])), 'y':torch.Tensor(test_data[user]['y']).long()} for user in test_data.keys()}
        condition_dim = num_classes if args.conditional else None
        data_dim = [3,64,64]
        latent_dim = args.latentdim
        img_size = data_dim
        manifold_data_vector_len = product(data_dim)

    return dataset_clients_train, dataset_clients_test, condition_dim, data_dim, latent_dim, num_classes, img_size, manifold_data_vector_len



def noniid(dataset, num_users, shard_per_user, num_classes, rand_set_all=[], testb=False):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    count = 0
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1

    shard_per_class = int(shard_per_user * num_users / num_classes)
    samples_per_user = int( count/num_users )
    # whether to sample more test samples per user
    double=False

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        if double:
            rand_set_label = list(rand_set_all[i]) * 50
        else:
            rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            try:
                idx = np.random.choice(len(idxs_dict[label]), replace=False)
            except:
                import pdb;pdb.set_trace()
            if (samples_per_user < 100 and testb):
                rand_set.append(idxs_dict[label][idx])
            else:
                rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        test.append(value)
    test = np.concatenate(test)
    all_len = [len(dict_users[k]) for k in dict_users.keys()]

    return dict_users, rand_set_all


def user_idx_to_dataset(dict_users, dataset, fx=None):
    dataset_dict = {}
    for user in dict_users.keys():
        x_all = []
        y_all = []
        for ind in dict_users[user]:
            x, y = dataset[ind]
            if fx:
                x = fx(x)
            x_all.append(x)
            y_all.append(y)

        dataset_dict[user] = {'x':torch.stack(x_all, dim=0), 'y':torch.Tensor(y_all).long()}

    return dataset_dict
