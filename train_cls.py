import json
import os
import configargparse
import glog as logger

from utils.tools import setup_seed, set_log_file, print_args, create_filename, args_conflict, str2bool
from dataset.dataset_utils import get_cls_dataset
from training_cls.trainer import FedTrainer

def parse_args():
    """ Parses command line arguments for the training """

    parser = configargparse.ArgumentParser()


    parser.add_argument("--modelname", type=str, default="cls", help="model name")
    parser.add_argument("--conditional", type=str2bool, default=True, help="whether to condition the label")

    # Dataset details
    parser.add_argument("--dataset", type=str, default='femnist', help="dataset name")
    parser.add_argument("--latentdim", type=int, default=-1, help="Manifold dimensionality (for datasets where that is variable)")
    parser.add_argument("--datadim", type=int, default=-1, help="True data dimensionality (for datasets where that is variable)")
    parser.add_argument("--num_classes", type=int, default=-1, help="classes num")
    parser.add_argument("--num_users", type=int, default=96, help="number of clients")
    parser.add_argument("--shard_per_user", type=int, default=5, help="number of clients")
    parser.add_argument("--user_frac", type=float, default=0.1, help="fraction of number of clients each epoch")
    # Synthetic dataset
    parser.add_argument("--data_var_dim", type=int, default=1, help="data dimensionality which varies")
    parser.add_argument("--data_cons_dim", type=int, default=1, help="data dimensionality which staies constant")
    parser.add_argument("--trainN", type=int, default=6, help="the number of generated train samples")
    parser.add_argument('--std', type=float, default=0.001, help="std")
    parser.add_argument('--sigma', type=float, default=0.1, help="data noise")
    parser.add_argument('--test_noise', type=str2bool, default=False, help="whether to add data noise to test data")
    parser.add_argument("--cls_hidden_features", type=int, default=8, help="hidden features of classifier")
    parser.add_argument("--cls_hidden_layer", type=int, default=4, help="number of hidden layer of classifier")

    # Manifold Model details
    parser.add_argument("--outerlayers", type=int, default=5, help="Number of transformations in f (not counting linear transformations)")
    parser.add_argument("--innerlayers", type=int, default=5, help="Number of transformations in h (not counting linear transformations)")
    parser.add_argument("--levels", type=int, default=4, help="Number of levels for multi-scale transform")
    parser.add_argument("--hidden_features", type=int, default=100, help="hidden features of normalizing flow")
    parser.add_argument("--coupling_type", type=str, default='affine', help="coupling_type, [rational_quadratic_spline, affine]")
    parser.add_argument("--dropout", type=float, default=0.0, help="Use dropout")
    parser.add_argument("--splinerange", default=10.0, type=float, help="Spline boundaries, only used for rq coupling layer")
    parser.add_argument("--splinebins", default=11, type=int, help="Number of spline bins, only used for rq coupling layer")
    parser.add_argument("--batchnorm", type=str2bool, default=False, help="Use batchnorm in ResNets")
    parser.add_argument("--outer_condition", type=str2bool, default=True, help="whether to apply condition on outer transform")
    parser.add_argument("--LU_linear", type=str2bool, default=True, help="whether to use LU linear")
    parser.add_argument("--outer_image", type=str2bool, default=False, help="whether to use image transform for outer transform")
    parser.add_argument("--intermediatensf", type=str2bool, default=False, help="Use NSF rather than linear layers before projecting (for M-flows and PIE on image data)")
    parser.add_argument("--linlayers", type=int, default=2, help="Number of linear layers before the projection for M-flow and PIE on image data")
    parser.add_argument("--linchannelfactor", type=int, default=2, help="Determines number of channels in linear trfs before the projection for M-flow and PIE on image data")
    parser.add_argument("--actnorm", type=str2bool, default=True, help="Use actnorm in convolutional architecture")

    # FedGen method
    parser.add_argument("--manifold_root", type=str, default='./', help="pth of manifold model")
    parser.add_argument("--manifold_pth", type=str, default='experiments_manifold', help="pth of manifold model")
    parser.add_argument("--latent_in_u", type=str2bool, default=True, help="where to extract latent data")
    parser.add_argument("--gen_share_each_label", type=str2bool, default=True, help="whether to generate/share in closure of each label. for femnist, must be True")
    # sample
    parser.add_argument("--generate_data_ratio", type=float, default=0.5, help="ratio of generated data number to original data num")
    parser.add_argument("--generate_data_weight", type=float, default=0.1, help="weight of generated data in loss")
    parser.add_argument("--gen_every_localepoch", type=str2bool, default=False, help="whether to generate every local epoch")
    parser.add_argument("--gen_within_closure", type=str2bool, default=True, help="whether to generate latent vectors within the closure of latent vectors of each client")
    parser.add_argument("--generate_data_prior", type=str, default='normal', help="type of data generation, specific to gen_within_closure, [uniform, normal]")
    # share
    parser.add_argument("--share_data", type=str2bool, default=True, help="whether to share data.")
    parser.add_argument("--closure_rate", type=float, default=1.0, help="shrink or expand the closure")
    parser.add_argument("--share_data_weight", type=float, default=0.1, help="weight of generated data in loss")


    # Training
    parser.add_argument("--fed_alg", type=str, default='local_gen', help="training algorithm, [local, fedavg, fedrep, local_gen, fedrep_gen, fedavg_gen]")
    parser.add_argument("--epochs", type=int, default=200, help="epochs")
    parser.add_argument("--local_epochs", type=int, default=15, help="local_epochs")
    parser.add_argument("--test_freq", type=int, default=10, help="epoch frequency for testing")
    parser.add_argument("--save_every", type=int, default=10, help="epoch frequency for saving model")
    parser.add_argument("--batchsize", type=int, default=10, help="Batch size")
    parser.add_argument("--manifold_batchsize", type=int, default=2000, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-2, help="Initial learning rate")
    parser.add_argument("--weightdecay", type=float, default=1.0e-5, help="Weight decay")
    parser.add_argument("--clip", type=float, default=10.0, help="Gradient norm clipping parameter")
    parser.add_argument("--seed", type=int, default=1357, help="Random seed (--i is always added to it)")
    parser.add_argument("--gpu", type=int, default=0, help="gpu index to use")
    parser.add_argument("--head_epochs", type=int, default=10, help="head epochs for local training, fedrep arg")
    parser.add_argument("--cls_dropout", type=float, default=0.6, help="drop out rate for classifying model")
    parser.add_argument("--cls_checkpoint", type=str, default=None, help="pth of the cls checkpoint")


    # Other settings
    parser.add_argument("--ssh", type=str2bool, default=False, help="whether is run by search")
    parser.add_argument("--debug", action="store_true", help="debug or not")
    parser.add_argument("--target_dir", type=str, default="experiments_cls", help="model name")
    parser.add_argument("--data_pth", type=str, default="/data", help="Base directory of data")
    parser.add_argument("--synthetic_data_pth", type=str, default="dataset_data/synthetic.npy", help="Base directory of data")

    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()
    setup_seed(args.seed)
    if not args.debug:
        set_log_file(create_filename('log_file', None, args), file_only=args.ssh)
    else:
        logger.info('------------------Debug--------------------')
    logger.setLevel('DEBUG')

    print_args(args)
    with open(create_filename('args', None, args), "w") as f:
        json.dump(vars(args), f, indent = 2)

    args_conflict(args)

    dataset_clients_train, dataset_clients_test, condition_dim, data_dim, latent_dim, num_classes, img_size, manifold_data_vector_len = get_cls_dataset(args)
    args.img_size = img_size
    args.latentdim = latent_dim
    args.datadim = data_dim
    args.num_classes = num_classes
    args.manifold_data_vector_len = manifold_data_vector_len

    trainer = FedTrainer(args, dataset_clients_train, dataset_clients_test, args.fed_alg, condition_dim)
    trainer.train(args)

