import os
import sys 
sys.path.append('../AWLoss')

import torch
from torchvision.transforms import Compose, Resize, Lambda, Normalize
from torch.utils.data import DataLoader, Subset
from monai.networks.nets import UNet

from examples.networks import *
from awloss import AWLoss
from examples.datasets import MaskedUltrasoundDataset2D
from examples.landscape import *


import matplotlib.pyplot as plt
import matplotlib.colors as clt
import progressbar
import random
import numpy as np

from saving import *


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = True

    return True


def set_device(device="cpu", idx=0):
    if device != "cpu":
        if torch.cuda.device_count() > idx and torch.cuda.is_available():
            print("Cuda installed! Running on GPU {} {}!".format(idx, torch.cuda.get_device_name(idx)))
            device="cuda:{}".format(idx)
        elif torch.cuda.device_count() > 0 and torch.cuda.is_available():
            print("Cuda installed but only {} GPU(s) available! Running on GPU 0 {}!".format(torch.cuda.device_count(), torch.cuda.get_device_name()))
            device="cuda:0"
        else:
            device="cpu"
            print("No GPU available! Running on CPU")
    return device


def get_params_to_update(model):
    """ Returns list of model parameters that have required_grad=True"""
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    return params_to_update


def print_single_stats(key, val):
    print(" %-45s %-15s %15s"%(key, ":", val))
    return None


def scale2range(x, range=[-1, 1]):
    return (x - x.min()) * (max(range) - min(range)) / (x.max() - x.min()) + min(range)

def clip_outliers(x, fence="outer"):
    f = 1.5 if fence=="inner" else 3.0
    q1 = torch.quantile(x, q=0.25)
    q3 = torch.quantile(x, q=0.75)
    iqr = q3 - q1
    lower = q1 - f*iqr
    upper = q3 + f*iqr
    x[torch.where(x < lower)] = lower
    x[torch.where(x > upper)] = upper
    return x

def create_mask(size, width, spacing):
    assert len(size) == len(width) == len(spacing)
    m = torch.ones(size)
    idxs = []
    for i in range(len(size)):
        wd, sp, s, = width[i], spacing[i], size[i]
        idx = []
        if wd > 0:
            for j in range(int(s / (wd + sp)) + 1):
                idx+= [j*(wd+sp) + k for k in range(wd)]
        else:
            idx += [j for j in range(s)]
        idx = [k for k in idx if k < s]
        idxs.append(torch.tensor(idx))
    idmesh = torch.meshgrid(*idxs)
    m[idmesh] = 0.
    return m
   
def train(model, train_loader, optimizer, criterion, scheduler=None, device="cpu"):
    """ Trains one epoch """
    model.train()
    
    total_loss = 0.

    for i , (X, target) in enumerate(train_loader):
        X, target = X.to(device), target.to(device)
        optimizer.zero_grad()

        # Forward pass
        recon = torch.sigmoid(model(X))

        # Evaluate losses
        loss  = criterion(recon, target)

        # Backprop and optim step
        loss.backward()
        optimizer.step()  
        
        # Keep track of total losses
        total_loss += loss / len(train_loader)
    
    if scheduler is not None:
        scheduler.step()
    
    return total_loss

def validate(model, train_loader, criterion, device="cpu"):
    """ Validates model with criterion """
    model.eval()

    total_loss = 0.

    for i , (X, target) in enumerate(train_loader):
        X, target = X.to(device), target.to(device)

        # Forward pass
        recon = torch.sigmoid(model(X))

        # Evaluate losses
        loss  = criterion(recon, target)
        
        # Keep track of total losses
        total_loss += loss / len(train_loader)
    
    return total_loss

def plot_losses(losses={}, filters={}, title=""):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for label, loss in losses.items():
        if len(loss) > 1:
            axs[0].plot(loss, label=label)
            axs[0].legend()
            axs[0].set_title(title)
            axs[0].set_xlabel("epoch")

    for name, v in filters.items():
        if len(v) > 1:
            axs[1].plot(v, label=name, alpha=0.5)
            axs[1].set_ylim(None, 1.1)
            axs[1].legend()
    plt.show()
    return fig

def plot_samples(samples={}, vmin=None, vmax=None):
    fig, axs = plt.subplots(1, len(samples), figsize=(15, 8))
    for i, (name, x) in enumerate(samples.items()):
        im = axs[i].imshow(x, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axs[i], shrink=0.5, norm = clt.Normalize(vmin=-1, vmax=1))
        axs[i].set_title(name)
    plt.show()
    return fig