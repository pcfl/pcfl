import random
from typing import DefaultDict
import numpy as np
import os
import glog as logger

def load_synthetic_data(synthetic_data_pth):
    data = np.load(synthetic_data_pth, allow_pickle=True).item()
    dataset_train = data['dataset_train']
    dataset_test = data['dataset_test']
    dict_users_train = data['dict_users_train']
    dict_users_test = data['dict_users_test']
    u_m_dict = data['u_m_dict']
    logger.info('Load synthetic data from [%s] instead of generating'%synthetic_data_pth)
    return dataset_train, dataset_test, dict_users_train, dict_users_test, u_m_dict

def generate_sinsynthetic_data(args):
    if os.path.exists(args.synthetic_data_pth):
        return load_synthetic_data(args.synthetic_data_pth)

    logger.info('Sin Synthetic data in [%s] not exists! Generating'%args.synthetic_data_pth)

    assert args.data_var_dim==1 and args.data_cons_dim==1

    args.testN = 30

    x_parts = [[np.pi/2*i, np.pi/2*(i+1)] for i in range(4)]

    dataset_train = []
    dataset_test = []
    dict_users_train = DefaultDict(list)
    dict_users_test = DefaultDict(list)

    u_m_dict = {}
    now_ind_train = 0
    now_ind_test = 0
    part_selected_times = np.zeros([len(x_parts)])
    for usr in range(args.num_users):
        part_num = 2
        parts_selected = [x_parts[usr%4], x_parts[(usr+1)%4]]
        u_m_dict[usr] = parts_selected
        for part in parts_selected:
            part_selected_times[x_parts.index(part)] += 1

            tmp_trainN =  args.trainN//part_num
            tmp_testN =  args.testN//part_num

            x_m_var_train = np.random.uniform(part[0], part[1], (tmp_trainN, args.data_var_dim))
            x_m_var_test = np.linspace(start=part[0], stop=part[1], num=tmp_testN)[:, None]
            x_m_var = np.concatenate([x_m_var_train, x_m_var_test], axis=0)
            x_m_cons = np.zeros([tmp_trainN+tmp_testN, args.data_cons_dim])
            x_m = np.concatenate((x_m_var, x_m_cons), axis=1)

            y_m = np.sin(x_m_var)[:,0]
            if usr>=(args.num_users//2):
                y_m = -y_m

            y_m[:tmp_trainN] = y_m[:tmp_trainN] + np.random.normal(0, args.sigma**2, (tmp_trainN,))
            if args.test_noise:
                y_m[tmp_trainN:] = y_m[tmp_trainN:] + np.random.normal(0, args.sigma**2, (tmp_testN,))

            dataset_train.extend([ np.concatenate([x.astype(np.float32),[y]]) for x, y in zip(x_m[:tmp_trainN], y_m[:tmp_trainN]) ])
            dataset_test.extend([ np.concatenate([x.astype(np.float32),[y]]) for x, y in zip(x_m[-tmp_testN:], y_m[-tmp_testN:]) ])

            dict_users_train[usr].extend([i+ now_ind_train for i in range(tmp_trainN)])
            dict_users_test[usr].extend([i+ now_ind_test for i in range(tmp_testN)])
            now_ind_train = now_ind_train + tmp_trainN
            now_ind_test = now_ind_test + tmp_testN

    dataset_train = np.array(dataset_train)
    dataset_test = np.array(dataset_test)
    os.makedirs(os.path.dirname(args.synthetic_data_pth), exist_ok=True)
    np.save(args.synthetic_data_pth, {'dataset_train':dataset_train, 'dataset_test':dataset_test,
                         'dict_users_train':dict_users_train, 'dict_users_test':dict_users_test, 'u_m_dict':u_m_dict})

    return dataset_train, dataset_test, dict_users_train, dict_users_test, u_m_dict
    