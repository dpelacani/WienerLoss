from losses import KLD
import numpy as np
import torch
import random
import torch.nn as nn
from pycm import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import progressbar

def train(model, train_loader, optimizer, criterion, device="cpu"):
    """ Trains one epoch"""
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
            recon = recon[0]
            # recon, mu, sigma = recon
            # kld_loss = kld(mu, sigma)

        # Evaluate losses
        loss  = criterion(recon, X)

        # Combining losses appropriately, backprop and take step
        combined_loss = loss.sum() + kld_loss.sum() 
        combined_loss.backward()
        optimizer.step()  
        
        # Keep track of total losses
        total_loss += loss.sum() / len(train_loader)
        total_kl   += kld_loss.sum() / len(train_loader) 
    
    return total_loss, total_kl

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
                # recon, mu, sigma = recon 

            # Evaluate losses
            loss  = criterion(recon, X)
            
            total_loss += loss.sum()/ len(train_loader)
    return total_loss


def train_model(model, optimizer, train_loader, loss, nepochs=150, log_frequency=10, sample=None, device="cpu", gradflow=False):
    print("\n\nTraining started ...")
    all_loss, all_mse, all_kl = [], [], []
    with progressbar.ProgressBar(max_value=nepochs) as bar:    
        for epoch in range(nepochs):
            epoch_loss, epoch_kl = train(model, train_loader, optimizer, loss, device=device)
            
            epoch_mse = validate(model, train_loader, nn.MSELoss(reduction="mean"), device=device)
            all_loss.append(epoch_loss.item())
            all_mse.append(epoch_mse.item())
            all_kl.append(epoch_kl.item())
            bar.update(epoch)
            
            # Metric logs
            log = {"epoch": epoch, "loss": epoch_loss.item(), "mse (validation)":epoch_mse.item(), "kl_loss": epoch_kl.item()}
            if (epoch % log_frequency == 0 or epoch==nepochs-1):
                print("\n", log)

            # Plots
            if sample is not None and (epoch % log_frequency == 0 or epoch==nepochs-1):
                # Model forward pass
                X = sample
                recon = model(X)
                if isinstance(recon, tuple): 
                    recon = recon[0] # in case model returns more than one output, recon must be the first
                
                # Loss evaluation and filters
                f = loss(recon, X)
                try:
                    v, T = loss.filters[0], loss.T
                    print(" argidx T, v: ",torch.argmax(torch.abs(T)).item(), torch.argmax(torch.abs(v)).item())
                except:
                    v, T = torch.tensor(0.), torch.tensor(0.)

                # Detach for plotting
                X, recon = X.cpu().detach().numpy(), recon.cpu().detach().numpy()
                v, T = v.cpu().detach().numpy(), T.cpu().detach().numpy()

                # Gradients
                if gradflow: plot_grad_flow(model.named_parameters())

                # Input, reconstructed, difference, filters and penalty
                if "AWLoss" in str(loss): 
                    if loss.filter_dim==1:
                        plot_train_sample1d(X, recon, v, T)
                    elif loss.filter_dim==2:
                         plot_train_sample2d(X, recon, v, np.repeat(np.expand_dims(T, 0), v.shape[0], 0))
                    elif loss.filter_dim==3:
                        plot_train_sample3d(X, recon, v, T)
                else:
                    # for losses that don't return filters
                    plot_train_sample(X, recon)

                # Plot losses
                fig, axs = plt.subplots(1,3, figsize=(7, 3))
                axs[0].plot(all_loss, label=str(loss))
                axs[0].legend()
                axs[0].set_xlabel("epoch")
                axs[1].plot(all_mse, label="mse avg")
                axs[1].legend()
                axs[1].set_xlabel("epoch")
                axs[2].plot(all_kl, label="kl_loss")
                axs[2].legend()
                axs[2].set_xlabel("epoch")
                plt.show()


def plot_train_sample(X, recon):
    X, recon = X[0].transpose(1,2,0), recon[0].transpose(1,2,0)

    fig, axs = plt.subplots(1,3)#, figsize=(15,15))
    axs[0].imshow(recon, cmap="gray", vmin=0., vmax=1.)
    axs[0].set_title("recon")

    axs[1].imshow(X, cmap="gray", vmin=0., vmax=1.)
    axs[1].set_title("orig")

    axs[2].imshow(X - recon, cmap="gray", vmin=-0.5, vmax=0.5)
    axs[2].set_title("diff")
    plt.show()
    return None


def plot_train_sample1d(X, recon, v, T):
    X, recon = X[0].transpose(1,2,0), recon[0].transpose(1,2,0)

    fig, axs = plt.subplots(2,3)
    axs[0, 0].imshow(recon, cmap="gray", vmin=0., vmax=1.)
    axs[0, 0].set_title("recon")

    axs[0, 1].imshow(X, cmap="gray", vmin=0., vmax=1.)
    axs[0, 1].set_title("orig")

    axs[0, 2].imshow(X - recon, cmap="gray", vmin=-0.5, vmax=0.5)
    axs[0, 2].set_title("diff")

    axs[1, 0].plot(v)
    axs[1, 0].plot(T, "--")
    axs[1, 0].set_title("T-1D vs v-1D")
    axs[1, 0].set_ylim(None, 1.1)

    axs[1, 1].plot((T.flatten() - v))
    axs[1, 1].set_title("T1D - v1D")
    axs[1, 1].set_ylim(None, 1.1)
    plt.show()
    return None

def plot_train_sample2d(X, recon, v, T):
    X, recon = X[0].transpose(1,2,0), recon[0].transpose(1,2,0)

    fig, axs = plt.subplots(2,3)#, figsize=(15,15))
    axs[0, 0].imshow(recon, cmap="gray", vmin=0, vmax=1.)
    axs[0, 0].set_title("recon")

    axs[0, 1].imshow(X, cmap="gray", vmin=0, vmax=1.)
    axs[0, 1].set_title("orig")

    axs[0, 2].imshow(X- recon, cmap="gray", vmin=-0.5, vmax=0.5)
    axs[0, 2].set_title("diff")

    try:
        axs[1, 0].imshow(T.transpose(1,2,0))
        axs[1, 0].set_title("T-2D")

        axs[1, 1].imshow(v.transpose(1,2,0))
        axs[1, 1].set_title("v-2D")

        axs[1, 2].plot((T.flatten() - v.flatten()))
        axs[1, 2].set_ylim(None, 1.1)
        axs[1, 2].set_title("T2D - v2D")
    except:
        pass
    plt.show()
    return None

def plot_train_sample3d(X, recon, v, T):
    X, recon = X[0].transpose(1,2,0), recon[0].transpose(1,2,0)

    fig, axs = plt.subplots(2,3)#, figsize=(15,15))
    axs[0, 0].imshow(recon, cmap="gray", vmin=0, vmax=1.)
    axs[0, 0].set_title("recon")

    axs[0, 1].imshow(X, cmap="gray", vmin=0, vmax=1.)
    axs[0, 1].set_title("orig")

    axs[0, 2].imshow(X- recon, cmap="gray", vmin=-0.5, vmax=0.5)
    axs[0, 2].set_title("diff")

    axs[1, 0].plot(T.flatten())
    axs[1, 0].set_title("T flattened")

    axs[1, 1].plot(v.flatten())
    axs[1, 1].set_title("v flattened")

    axs[1, 2].plot((T.flatten() - v.flatten()))
    axs[1, 2].set_ylim(None, 1.1)
    axs[1, 2].set_title("T - v")

    plt.show()
    return None

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