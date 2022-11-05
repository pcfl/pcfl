import json
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import glog as logger
import numpy as np

def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(train_data.keys())

    return clients, groups, train_data, test_data


class All_data_json(Dataset):
    def __init__(self, data_pth, num_class, train_flag, feature_transform=None, labels_transform=None):
        super().__init__()
        self.num_class = num_class

        clients, _, train_data, test_data = read_data(os.path.join(data_pth, 'train'), os.path.join(data_pth, 'test'))
        data = train_data if train_flag else test_data
        features = []
        labels = []
        for k,v in data.items():
            features += v['x']
            labels += v['y']
        
        if feature_transform:
            features = feature_transform(features)
        
        if labels_transform:
            labels = labels_transform(labels)

        self.features = torch.Tensor(features)
        self.labels = torch.Tensor(labels).long()

        logger.info('Data len: %s, train: %s'%(str(self.features.shape), str(train_flag)))

    def __getitem__(self, index):
        x = self.features[index]
        x = x + (torch.rand_like(x)/255.0)
        label_one_hot = F.one_hot(self.labels[index], num_classes=self.num_class)
        return x, label_one_hot
    
    def __len__(self):
        return self.features.shape[0]

class Adult_NF_dataset(Dataset):
    def __init__(self, data_pth, num_class, train_flag, concat_label):
        super().__init__()
        self.num_class = num_class

        with open(data_pth, "r") as f:
            data = json.load(f)["user_data"]
        self.features = torch.stack([torch.Tensor(x) for x in data["phd"]["x"]] + [torch.Tensor(x) for x in data["non-phd"]["x"]], dim=0)
        self.labels = torch.stack([torch.Tensor([y]).long() for y in data["phd"]["y"]] + [torch.Tensor([y]).long() for y in data["non-phd"]["y"]], dim=0)[:,0]

        if concat_label:
            self.features = torch.cat([self.features, self.labels.float().unsqueeze(1)], dim=1)

        logger.info('Data len: %s, train: %s'%(str(self.features.shape), str(train_flag)))
        
    def __getitem__(self, index):
        x = self.features[index]
        x = x + (torch.rand_like(x)/2)
        label_one_hot = F.one_hot(self.labels[index], num_classes=self.num_class)
        return x, label_one_hot
    
    def __len__(self):
        return self.features.shape[0]