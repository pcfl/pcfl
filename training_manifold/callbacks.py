# Reference to https://github.com/johannbrehmer/manifold-flow

import torch
import os
import numpy as np
import logging
import random
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def save_imgs(dirname, img_size, imgs):
    os.makedirs(dirname)
    for i, img in enumerate(imgs):
        x = np.clip(np.reshape(img.detach().cpu().numpy(), [*img_size]), 0.0, 1.0)
        plt.figure()
        plt.imshow(x)
        plt.savefig(os.path.join(dirname, '%03d.jpg'%i))
        plt.close()

def save_model_after_every_epoch(filename):
    """ Saves model checkpoints. """

    def callback(i_epoch, model, loss_train, loss_val, subset=None, trainer=None, last_batch=None):
        if i_epoch < 0:
            return

        torch.save(model.state_dict(), filename.format("last"))
        if (i_epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), filename.format(i_epoch + 1))

    return callback

def plot_sample_images(filename, img_size, device, f_label_toname, context=None):
    """ Saves model checkpoints. """

    def callback(i_epoch, model, loss_train, loss_val, subset=None, trainer=None, last_batch=None):
        if i_epoch in [0, 1, 4] or (i_epoch + 1) % 5 == 0:
            if not type(context)==type(None):
                context_ = torch.Tensor(np.array(random.choices(context, k=30))).to(device)
            else:
                context_ = None
            x = model.sample(n=30, context=context_).detach().cpu().numpy()
            x = np.clip(np.reshape(x, [-1, *img_size]), 0.0, 1.0)
            x = np.transpose(x, [0, 2, 3, 1])

            plt.figure(figsize=(6 * 3.0, 5 * 3.0))
            for i in range(30):
                ax = plt.subplot(5, 6, i + 1)
                if context_ != None:
                    num = context_[i].argmax().item()
                    name = f_label_toname(num)
                    ax.set_title(name, fontsize = 30, fontweight="bold")
                plt.imshow(x[i])
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)
            plt.tight_layout()
            plt.savefig(filename.format(i_epoch + 1))
            plt.close()

    return callback

def plot_sample_images_temperatures(filename, img_size, device, f_label_toname, context=None):
    """ Saves model checkpoints. """
    temperatures = [0., 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    n_each = 5

    def callback(i_epoch, model, loss_train, loss_val, subset=None, trainer=None, last_batch=None):
        if i_epoch in [0, 1, 4] or (i_epoch + 1) % 5 == 0:
            if not type(context)==type(None):
                context_ = [torch.Tensor(np.array(random.choices(context, k=n_each))).to(device) for t in temperatures]
            else:
                context_ = [None for t in temperatures]
            
            
            plt.figure(figsize=(len(temperatures)*2., n_each*2.))
            us = [t * torch.randn(n_each, model.latent_dim, dtype=torch.float).to(device) for t in temperatures]
            xs = [model.sample(u=u, n=n_each, sample_orthogonal=False, context=context_[ui]).detach().cpu().numpy() for ui, u in enumerate(us)]
            for i in range(len(xs)):
                for j in range(n_each):
                    x = np.clip(np.reshape(xs[i][j], [*img_size]), 0.0, 1.0)
                    x = np.transpose(x, [1, 2, 0])
                    ax = plt.subplot(n_each, len(temperatures), j*len(temperatures) + i + 1)
                    plt.imshow(x)
                    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

                    title = ''
                    if j == 0:
                        title += f"T = {temperatures[i]}"
                    if context_[i]!=None:
                        num = context_[i][j].argmax().item()
                        title += '\n'+f_label_toname(num)
                    if title!='':
                        plt.title(title, fontweight="bold")

            plt.tight_layout()
            plt.savefig(filename.format(i_epoch + 1))
            plt.close()

    return callback

def plot_reco_images(filename, img_size):
    """ Saves model checkpoints. """

    def callback(i_epoch, model, loss_train, loss_val, subset=None, trainer=None, last_batch=None):
        if last_batch is None:
            return

        if i_epoch in [0, 1, 4] or (i_epoch + 1) % 5 == 0:
            x = last_batch["x"]
            x_reco = last_batch["x_reco"]

            x = np.clip(np.reshape(x, [-1, *img_size]), 0.0, 1.0)
            x_reco = np.clip(np.reshape(x_reco, [-1, *img_size]), 0.0, 1.0)
            x = np.transpose(x, [0, 2, 3, 1])
            x_reco = np.transpose(x_reco, [0, 2, 3, 1])

            plt.figure(figsize=(6 * 3.0, 5 * 3.0))
            for i in range(15):
                plt.subplot(5, 6, 2 * i + 1)
                plt.imshow(x[i])
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)

                plt.subplot(5, 6, 2 * i + 2)
                plt.imshow(x_reco[i])
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)

            plt.tight_layout()
            plt.savefig(filename.format(i_epoch + 1))
            plt.close()

    return callback


def print_mf_latent_statistics():
    """ Prints debug info about size of weights. """

    def callback(i_epoch, model, loss_train, loss_val, subset=None, trainer=None, last_batch=None):
        if last_batch is None:
            return

        u = last_batch["u"]

        logger.debug(f"           Latent variables: mean = {np.mean(u):>8.5f}")
        logger.debug(f"                             std  = {np.std(u):>8.5f}")
        logger.debug(f"                             min  = {np.min(u):>8.5f}")
        logger.debug(f"                             max  = {np.max(u):>8.5f}")

    return callback


def print_mf_weight_statistics():
    """ Prints debug info about size of weights. """

    def callback(i_epoch, model, loss_train, loss_val, subset=None, trainer=None, last_batch=None):
        models, labels = [], []
        try:
            models.append(model.outer_transform)
            labels.append("outer transform weights:")
        except:
            pass
        try:
            models.append(model.inner_transform)
            labels.append("inner transform weights:")
        except:
            pass
        try:
            models.append(model.transform)
            labels.append("transform weights:")
        except:
            pass
        try:
            models.append(model.encoder)
            labels.append("encoder weights:")
        except:
            pass
        try:
            models.append(model.decoder)
            labels.append("decoder weights:")
        except:
            pass

        subset_str = "          " if subset is None or trainer is None else "  {:>2d} / {:>2d}:".format(subset, trainer)

        for model_, label_ in zip(models, labels):
            weights = np.hstack([param.detach().cpu().numpy().flatten() for param in model_.parameters()])
            logger.debug(
                "{} {:26.26s} mean {:>8.5f}, std {:>8.5f}, range {:>8.5f} ... {:>8.5f}".format(
                    subset_str, label_, np.mean(weights), np.std(weights), np.min(weights), np.max(weights)
                )
            )

    return callback
