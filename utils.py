from losses import KLD
import random
import numpy as np
import torch
import random
import torch.nn as nn
from pycm import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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


def validate(model, train_loader, criterion, device="cpu"):
    """ Validates loss of a data loader of an autoencoder or variational autoencoder whose output has the recon image as the first item """
    model.eval()
    total_loss = 0.

    with torch.no_grad():
        for i , (X, _) in enumerate(train_loader):
            X = X.to(device)
            
            # Reconstructed images from forward pass
            recon = model(X)
            if isinstance(recon, tuple): # if model returns more than one variable
                recon = recon[0] 

            # Evaluate losses
            loss =  criterion(recon, X)

            total_loss += loss / len(train_loader)
    return total_loss


def train(model, train_loader, optimizer, criterion, device="cpu"):
    """ Trains one epoch of"""
    model.train()
    
    kld = KLD()
    total_loss, total_kl = 0., 0.,

    for i , (X, _) in enumerate(train_loader):
        X = X.to(device)
        optimizer.zero_grad()

        # Reconstructed images from forward pass
        recon = model(X)
        kld_loss = torch.tensor([0.]).to(device)
        if isinstance(recon, tuple):
            recon, mu, sigma = recon
            kld_loss = kld(mu, sigma)

        # Evaluate losses
        loss =  criterion(recon, X)

        # Combining losses appropriately, backprop and take step
        combined_loss = loss + kld_loss 
        combined_loss.backward()
        optimizer.step()  
        
        # Keep track of total losses
        total_loss += loss / len(train_loader)
        total_kl   += kld_loss / len(train_loader) 
        
    return total_loss, total_kl


def plot_grad_flow(named_parameters):
    '''
    https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10

    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.3, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()
    return None

