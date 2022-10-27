import os
import json
import pathlib
import sched
import dill
import pickle

import torch
import numpy as np
import progressbar
from torchvision.utils import make_grid

from pycm import *
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


def save_pickle(obj, filename):
    with open(filename, 'wb') as f:  # Overwrites any existing file.
        dill.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, 'rb') as f:  # Overwrites any existing file.
        try:
            obj = dill.load(f)
        except RuntimeError:
            obj = torch.load(f, map_location=torch.device('cpu'))
    return obj


def save_exp(objs={}, figs={}, summary="", path=None, overwrite=False, verbose=False):
    # Create folder to save experiment
    if path is None:
        cnt = 0
        path = pathlib.Path("./exps/exp%g/"%cnt)
        if not overwrite:
            while path.exists():
                cnt += 1
                path = pathlib.Path("./exps/exp%g"%cnt)
    else:
        path = pathlib.Path(path)

    assert (not path.exists() or overwrite), "%s already exists"%path.resolve()
    try:
        os.mkdir(path)
    except:
        pass

    if verbose:
        print("Saving experiment at %s ..."%path.resolve())

    for name, obj in objs.items():
        filename = path / pathlib.Path("%s.pkl"%name)
        save_pickle(obj, filename.resolve())
        if verbose:
            print("\t %s"%filename)

    for name, fig in figs.items():
        filename = path / pathlib.Path("%s.png"%name)
        fig.savefig(filename, facecolor='w', transparent=False)
        if verbose:
            print("\t %s"%filename)

    with open(path / pathlib.Path("summary.json"), 'w') as f:
        summary["exp"] = path.parts[-1]
        json.dump(summary, f) 
        if verbose:
            print("\t summary.json")
    return path


def load_exp(path):
    exp = {}
    filepaths = pathlib.Path(path).glob('**/*')
    for p in filepaths:
        if p.suffix == ".pkl":
            name = p.with_suffix('').parts[-1]
            exp[name] = load_pickle(p)
    return exp


def train(model, dataloader, optimizer, criterion, scheduler=None, device="cpu"):
    """ Trains one epoch of a dataloader of an autoencoder """
    model.train()
    
    total_loss, total_kl = 0., 0.,

    for i , (X, _) in enumerate(dataloader):
        X = X.to(device)
        optimizer.zero_grad()

        # Reconstructed images from forward pass
        recon = model(X)

        # Evaluate losses
        loss  = criterion(recon, X)

        # Backprop and take step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  
        
        # Keep track of total losses
        total_loss += loss.sum() / len(dataloader)

    if scheduler:
        scheduler.step()
    
    return total_loss

def validate(model, dataloader, criterion, device="cpu"):
    """ Validates loss of a data loader of an autoencoder """
    model.eval()
    total_loss = 0.

    with torch.no_grad():
        for i , (X, _) in enumerate(dataloader):
            X = X.to(device)
            
            # Reconstructed images from forward pass
            recon = model(X)

            # Evaluate losses
            loss  = criterion(recon, X)
            
            total_loss += loss.sum()/ len(dataloader)
    return total_loss


def train_model(model, optimizer, train_loader, valid_loader, loss, nepochs=150, log_frequency=10, 
                scheduler=None, device="cpu", gradflow=False, save=False, fcmap="seismic", summary_app={}):

    # Sample for visualisation:
    try:
        sample_batch = next(iter(valid_loader))[0][:32].to(device)
    except:
        sample_batch = next(iter(train_loader))[0][:32].to(device)

    # Training loop
    print("\n\nTraining started ...")
    all_train_losses, all_valid_losses = [], []
    with progressbar.ProgressBar(max_value=nepochs) as bar:    
        for epoch in range(nepochs):
            bar.update(epoch)

            # Train and validate epoch
            train_loss = train(model, train_loader, optimizer, loss, scheduler, device)
            all_train_losses.append(train_loss.item())
            if valid_loader:
                valid_loss = validate(model, valid_loader, loss, device)
                all_valid_losses.append(valid_loss.item())
            
            # Logging
            log = {"epoch": epoch, "train_loss": train_loss.item()}
            if valid_loader:
                log.update({"valid_loss": valid_loss.item()})

            if (epoch == 0 or epoch % log_frequency == 0 or epoch==nepochs-1):
                print("\n", log)

                # Show batch reconstruction and filters
                model.eval()
                recon = model(sample_batch)
                _ = loss(recon, sample_batch)
                try:
                    v = loss.filters
                    print("Filters range: [{:.2f} , {:.2f}]".format(v.min(), v.max()))
                except:
                    v = torch.zeros_like(recon)

                sample_fig = show_grid(sample_batch=sample_batch, recon=recon, filters=v,
                            figsize=(5*sample_batch.shape[0], 15))
                losses_fig = plot_losses(losses={"train": all_train_losses, "valid":all_valid_losses})
                
                if gradflow:
                    grad_fig = plot_grad_flow(model.named_parameters())


                # Prepare dictionaries for saving
                if save:
                    figs = {"losses":losses_fig, "samples":sample_fig}       # Saves png
                    if gradflow: figs.update({"gradflow": grad_fig})
                    
                    objs = {"train_loader":train_loader,                     # Saves pkl
                        "valid_loader":valid_loader,
                        "sample_batch": sample_batch,
                        "recon": recon,
                        "model": model,
                        "optim": optimizer,
                        "loss": loss,
                        "train_losses": all_train_losses,
                        "vald_losses": all_valid_losses,
                        }

                    summary = {"model_name": type(model.module).__name__,     # Saves json
                        "loss": str(loss),
                        "img_size": sample_batch[0].detach().cpu().numpy().shape,
                        "device":device,
                        "nepochs": nepochs,
                        "current_epoch":epoch, 
                        "learning_rate":optimizer.defaults["lr"],
                        "batch_size":train_loader.batch_size,
                        "ntrain": len(train_loader.dataset),
                        "nvalid": len(valid_loader.dataset) if valid_loader is not None else None}
                    summary.update(summary_app)

                    # Save
                    if epoch == 0:
                        p = save_exp(objs=objs, figs=figs, summary=summary, 
                                path=None, overwrite=False)
                    else:
                        p = save_exp(objs=objs, figs=figs, summary=summary, 
                                path=p, overwrite=True)


def scale2range(x, range=[-1, 1]):
    return (x - x.min()) * (max(range) - min(range)) / (x.max() - x.min()) + min(range)            


def show_grid(sample_batch, recon, filters, figsize=(10, 15), fcmap="seismic"):
    fig, axs = plt.subplots(3, 1, figsize=figsize)
    axs[0].imshow(make_grid(recon, pad_value=0, padding=2, vmin=0, vmax=1).cpu().data.permute(1, 2, 0))
    axs[1].imshow(make_grid(sample_batch, pad_value=0, padding=2, vmin=0, vmax=1).cpu().data.permute(1, 2, 0))
    axs[2].imshow(make_grid(filters, pad_value=0, padding=2).cpu().data[0], vmin=-0.1, vmax=0.1, cmap=fcmap)
    
    axs[0].set_title("Reconstruction")
    axs[1].set_title("Target")
    axs[2].set_title("Filters Channel 0")
    plt.show()
    return fig

def plot_losses(losses={}, title=""):
    fig = plt.figure(figsize=(6, 6))
    for label, loss in losses.items():
        if len(loss) > 1:
            plt.plot(loss, label=label)
            plt.legend()
            plt.title(title)
            plt.xlabel("epoch")
    plt.show()
    return fig

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
            ave_grads.append(p.grad.abs().mean().detach().cpu())
            max_grads.append(p.grad.abs().max().detach().cpu())

    fig = plt.figure(figsize=(8, 5))
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
    return fig