import os
import sys
import numpy as np
import torch
import random
import glog as logger
import configargparse

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
    
def set_log_file(fname, file_only=False):
    # set log file
    # simple tricks for duplicating logging destination in the logging module such as:
    # logging.getLogger().addHandler(logging.FileHandler(filename))
    # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything
    if file_only:
        # we only output messages to file, and stdout/stderr receives nothing.
        # this feature is designed for executing the script via ssh:
        # since ssh has a windowing kind of flow control, i.e., if the controller does not read data from a
        # ssh channel and its buffer fills up, the execution machine will not be able to write anything into the
        # channel and the process will be set to sleeping (S) status until someone reads all data from the channel.
        # this is not desired since we do not want to read stdout/stderr from the controller machine.
        # so, here we use a simple solution: disable output to stdout/stderr and only output messages to log file.
        logger.logger.handlers[0].stream = logger.handler.stream = sys.stdout = sys.stderr = open(fname, 'w', buffering=1)
    else:
        # we output messages to both file and stdout/stderr
        import subprocess
        tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def print_args(args):
    logger.info('-------- args -----------')
    for k,v in vars(args).items():
        logger.info('%s: '%k+str(v))
    logger.info('-------------------------')

def create_filename(type_, label, args):

    if type_ == "checkpoint":
        filename = "{}/data/models/checkpoints/{}_{}_{}.pt".format(args.target_dir, args.modelname, "epoch" if label is None else "epoch_" + label, "{}")

    elif type_ == "model":
        filename = "{}/data/models/{}.pt".format(args.target_dir, args.modelname)

    elif type_ == "training_plot":
        filename = "{}/figures/training/{}_{}_{}.pdf".format(args.target_dir, args.modelname, "epoch" if label is None else label, "{}")

    elif type_ == "learning_curve":
        filename = "{}/data/learning_curves/{}.npy".format(args.target_dir, args.modelname)

    elif type_ == "log_file":
        filename = "{}/train_{}.log".format(args.target_dir, args.modelname)

    elif type_ == "cls_model":
        filename = "{}/models/{}.pth".format(args.target_dir, label)

    elif type_ == "args":
        filename = "{}/args.json".format(args.target_dir)
    
    elif type_ == "pickle":
        filename = "{}/pickle.pkl".format(args.target_dir)

    else:
        raise NotImplementedError

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    return filename

def nat_to_bit_per_dim(dim):
    if isinstance(dim, (tuple, list, np.ndarray)):
        dim = np.product(dim)
    logger.debug("Nat to bit per dim: factor %s", 1.0 / (np.log(2) * dim))
    return 1.0 / (np.log(2) * dim)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')

def args_conflict(args):
    if args.generate_data_ratio>0:
        if args.conditional:
            assert args.gen_share_each_label # if gen_within_closure, closure must be separately calculated for each class,
                                       # otherwise it would be meaningless
        else:
            assert args.gen_within_closure
    
    if args.share_data:
        if args.conditional:
            assert args.gen_share_each_label

    if args.dataset=='synthetic':
        assert (not args.conditional) and (args.gen_within_closure) and (not args.share_each_label)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')

def model_load(model_type, pth, device, cls_model):
    with open(pth, 'rb') as f:
        checkpoint = torch.load(f, map_location=device)

    if model_type == "global":
        cls_model.load_state_dict(checkpoint)
        logger.info("=> {} model load, checkpoint found at {}".format(model_type, pth))
        return cls_model
    elif model_type == "local":
        w_locals = checkpoint
        logger.info("=> {} model load, checkpoint found at {}".format(model_type, pth))
        return w_locals