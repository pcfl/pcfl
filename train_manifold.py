import os
import numpy as np
from torch.utils.data import DataLoader
import configargparse
import glog as logger
from torch import optim
import torch
import matplotlib.pyplot as plt

from dataset.dataset_utils import get_manifold_dataset
from manifold.manifold_model import ManifoldFlow
from training_manifold import ForwardTrainer, callbacks, losses
from utils.tools import setup_seed, set_log_file, create_filename, nat_to_bit_per_dim, print_args, str2bool

def parse_args():
    """ Parses command line arguments for the training """

    parser = configargparse.ArgumentParser()

    parser.add_argument("--ssh", type=str2bool, default=False, help="whether is run by search")
    parser.add_argument("--debug", action="store_true", default=False, help="debug or not")
    parser.add_argument("--gpu", type=int, default=0, help="gpu index to use")
    parser.add_argument("--modelname", type=str, default="manifold", help="model name")
    parser.add_argument("--target_dir", type=str, default="experiments_manifold", help="model name")
    parser.add_argument("--conditional", type=str2bool, default=True, help="whether to condition the label")

    # Dataset details
    parser.add_argument("--dataset", type=str, default='femnist', help="dataset name")
    parser.add_argument("--latentdim", type=int, default=32, help="Manifold dimensionality (for datasets where that is variable)")
    parser.add_argument("--datadim", type=int, default=-1, help="True data dimensionality (for datasets where that is variable)")
    parser.add_argument("--img_size", type=list, default=[], help="True data dimensionality (for datasets where that is variable)")
    # Synthetic dataset
    parser.add_argument("--data_var_dim", type=int, default=1, help="data dimensionality which varies")
    parser.add_argument("--data_cons_dim", type=int, default=1, help="data dimensionality which staies constant")
    parser.add_argument("--trainN", type=int, default=6, help="the number of generated train samples")
    parser.add_argument('--std', type=float, default=0.001, help="std")
    parser.add_argument('--sigma', type=float, default=0.1, help="learning rate for preference vector")

    # Model details
    parser.add_argument("--outerlayers", type=int, default=5, help="Number of transformations in f (not counting linear transformations)")
    parser.add_argument("--innerlayers", type=int, default=5, help="Number of transformations in h (not counting linear transformations)")
    parser.add_argument("--levels", type=int, default=4, help="Number of levels for multi-scale transform")
    parser.add_argument("--hidden_features", type=int, default=6, help="hidden features of normalizing flow")
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

    # Training
    parser.add_argument("--epoch_recon", type=int, default=200, help="Maximum number of epochs for reconstruction learning")
    parser.add_argument("--epoch_density", type=int, default=200, help="Maximum number of epochs for normalizing flow learning")
    parser.add_argument("--resume", type=int, default=None, help="resume epoch")
    parser.add_argument("--resume_pth", type=str, default='outputs_manifold/latent_dim/experiments_manifold_latent512_condition', help="resume model pth")
    parser.add_argument("--batchsize", type=int, default=50, help="Batch size for everything except OT training")
    parser.add_argument("--lr", type=float, default=1e-2, help="Initial learning rate")
    parser.add_argument("--msefactor", type=float, default=1000.0, help="Reco error multiplier in loss")
    parser.add_argument("--nllfactor", type=float, default=1.0, help="Negative log likelihood multiplier in loss (except for M-flow-S training)")
    parser.add_argument("--weightdecay", type=float, default=1.0e-5, help="Weight decay")
    parser.add_argument("--clip", type=float, default=10.0, help="Gradient norm clipping parameter")
    parser.add_argument("--l1", type=str2bool, default=False, help="Use smooth L1 loss rather than L2 (MSE) for reco error")
    parser.add_argument("--uvl2reg", type=float, default=0.01, help="Add L2 regularization term on the latent variables after the outer flow (M-flow-M/D only)")
    parser.add_argument("--seed", type=int, default=1357, help="Random seed (--i is always added to it)")

    # Federated Training
    parser.add_argument("--num_users", type=int, default=96, help="number of clients")

    # Other settings
    parser.add_argument("--data_pth", type=str, default="/data", help="Base directory of data")
    parser.add_argument("--synthetic_data_pth", type=str, default="dataset_data/synthetic.npy", help="Base directory of data")

    args = parser.parse_args()
    return args

def create_one_hot_context(context_dim):
    return np.eye(context_dim)

def train_manifold_flow_sequential(args, train_loader, test_loader, model, condition_dim, is_image, device, f_label_toname):
    """ Sequential M-flow-M/D training """

    trainer1 = ForwardTrainer(model, device, conditional=args.conditional)
    trainer2 = ForwardTrainer(model, device, conditional=args.conditional)

    common_kwargs = {
        "train_loader": train_loader,
        "val_loader": test_loader,
        "initial_lr": args.lr,
        "scheduler": optim.lr_scheduler.CosineAnnealingLR,
        "clip_gradient": args.clip,
    }
    if args.weightdecay is not None:
        common_kwargs["optimizer_kwargs"] = {"weight_decay": float(args.weightdecay)}

    callbacks1, callbacks2 = [], []
    if is_image:
        callbacks1.append(
            callbacks.plot_sample_images(
                create_filename("training_plot", "sample_epoch_A", args), img_size=args.img_size, device=device,
                context=create_one_hot_context(condition_dim) if args.conditional else None, f_label_toname=f_label_toname
            )
        )
        callbacks2.append(
            callbacks.plot_sample_images(
                create_filename("training_plot", "sample_epoch_B", args), img_size=args.img_size, device=device,
                context=create_one_hot_context(condition_dim) if args.conditional else None, f_label_toname=f_label_toname
            )
        )
        callbacks1.append(callbacks.plot_reco_images(create_filename("training_plot", "reco_epoch_A", args), img_size=args.img_size))
        callbacks2.append(callbacks.plot_reco_images(create_filename("training_plot", "reco_epoch_B", args), img_size=args.img_size))

    logger.info("Starting training MF, phase 1: manifold training")

    learning_curves = trainer1.train(
        loss_functions=[losses.smooth_l1_loss if args.l1 else losses.mse] + ([] if args.uvl2reg is None else [losses.hiddenl2reg]),
        loss_labels=["L1" if args.l1 else "MSE"] + ([] if args.uvl2reg is None else ["L2_lat"]),
        loss_weights=[args.msefactor] + ([] if args.uvl2reg is None else [args.uvl2reg]),
        epochs=args.epoch_recon,
        parameters=list(model.outer_transform.parameters()),
        callbacks=callbacks1,
        forward_kwargs={"mode": "projection", "return_hidden": args.uvl2reg is not None},
        initial_epoch=args.startepoch,
        seed =  args.seed + 1,
        **common_kwargs,
    )
    learning_curves = np.vstack(learning_curves).T
    
    logger.info("Starting training MF, phase 2: density training")
    learning_curves_ = trainer2.train(
        loss_functions=[losses.nll],
        loss_labels=["NLL"],
        loss_weights=[args.nllfactor * nat_to_bit_per_dim(args.latentdim)],
        epochs=args.epoch_density,
        parameters=list(model.inner_transform.parameters()),
        callbacks=callbacks2,
        forward_kwargs={"mode": "mf-fixed-manifold"},
        initial_epoch=args.startepoch-args.epoch_recon,
        seed =  args.seed + 2,
        **common_kwargs,
    )
    learning_curves = np.vstack((learning_curves, np.vstack(learning_curves_).T))

    return learning_curves

if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    setup_seed(args.seed)
    if not args.ssh:
        args.target_dir = args.target_dir + '_latent%d'%args.latentdim + ('_condition' if args.conditional else '_ncondition')
    if not args.debug:
        set_log_file(create_filename('log_file', None, args), file_only=args.ssh)
    else:
        logger.info('------------------Debug--------------------')
    logger.setLevel('DEBUG')

    plt.set_loglevel('error')
    print_args(args)

    dataset_train, dataset_test, condition_dim, img_size, data_dim, latent_dim, is_image, f_label_toname = get_manifold_dataset(args)
    train_loader = DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True, num_workers=6, drop_last=False)
    test_loader = DataLoader(dataset_test, batch_size=args.batchsize, shuffle=False, num_workers=6, drop_last=False)
    args.img_size = img_size
    args.latentdim = latent_dim
    args.datadim = data_dim

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    model = ManifoldFlow(device, args, data_vector_len=data_dim, latent_dim=latent_dim, pie_epsilon=1.0e-2,
                        condition_dim=condition_dim if args.conditional else None,
                         apply_context_to_outer=(args.outer_condition and args.conditional), clip_pie=False)
    logger.info(model)

    args.startepoch = 0
    if args.resume is not None:
        args.startepoch = args.resume
        model.load_state_dict(torch.load(os.path.join(args.resume_pth, 'data/models/manifold.pt'), map_location=torch.device("cpu")))

    # Train and save
    learning_curves = train_manifold_flow_sequential(args, train_loader, test_loader, model, condition_dim, is_image=is_image, device=device, f_label_toname=f_label_toname)

    # Save
    logger.info("Saving model")
    torch.save(model.state_dict(), create_filename("model", None, args))

    logger.info("All done! Have a nice day!")
