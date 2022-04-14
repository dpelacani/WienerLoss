import os
import sys 
sys.path.append('../../')

import torch
from torchvision.transforms import Compose, Resize, Lambda, Normalize
from torch.utils.data import DataLoader
from skimage.measure import compare_ssim, compare_mse
from sklearn.preprocessing import RobustScaler
from monai.networks.nets import UNet

from networks import *
from utils import set_device, set_seed
from losses import AWLoss
from datasets import MaskedUltrasoundDataset2D


import matplotlib.pyplot as plt
import progressbar
import plotly.express as px
import plotly.graph_objects as go


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

def create_mask(size, width):
    assert len(size) == len(width)
    m = torch.zeros(size)
    idxs = []
    for i in range(len(size)):
        dx = width[i]
        idx = [j*dx for j in range(int(size[i]/dx) +1)]
        idx = idx[:-1] if idx[-1] == size[i] else idx
        idxs.append(torch.tensor(idx))
    idmesh = torch.meshgrid(*idxs)
    
    m[idmesh] = 1.
    return m

def make_model(nc=64, device="cpu"):
    channels = (16, 32, 64, 128, 256)
    return UNet(
    spatial_dims=2,
    in_channels=nc,
    out_channels=nc,
    channels=channels,
    strides=tuple([2 for i in range(len(channels))]), 
    num_res_units=2,
    act="mish",
).to(device)

def train(model, train_loader, optimizer, criterion, wmse=0., device="cpu"):
    """ Trains one epoch"""
    model.train()
    
    total_loss = 0.
    total_mse = 0.

    for i , (X, target) in enumerate(train_loader):
        X, target = X.to(device), target.to(device)
        optimizer.zero_grad()

        # Forward pass
        recon = torch.tanh(model(X))

        # Evaluate losses
        loss  = criterion(recon, target)
        mse = nn.MSELoss(reduction="sum")(recon, target)
        combied_loss = loss + wmse*mse

        # Backprop and optim step
        combied_loss.backward()
        optimizer.step()  
        
        # Keep track of total losses
        total_loss += loss / len(train_loader)
        total_mse += mse / len(train_loader)
    
    return total_loss, total_mse

def validate(model, train_loader, criterion, device="cpu"):
    """ Validates loss of a data loader of an autoencoder """
    model.eval()
    total_loss = 0.

    with torch.no_grad():
        for i , (X, target) in enumerate(train_loader):
            X, target = X.to(device), target.to(device)
            
            # Forward Pass
            recon = torch.tanh(model(X))

            # Evaluate losses
            loss  = criterion(recon, target)
            
            total_loss += loss / len(train_loader)
    return total_loss


def train_model(model, optimizer, loss, train_loader, wmse=0., valid_loader=None, nepochs=150, log_frequency=10, sample_input=None, sample_target=None, device="cpu"):
    print("\n\nTraining started ...")
    train_losses, valid_losses, mse = [], [], []
    with progressbar.ProgressBar(max_value=nepochs) as bar:    
        for epoch in range(nepochs):
            # Train epoch
            epoch_loss, mse_loss = train(model, train_loader, optimizer, loss, wmse, device)
            # validate
            # ssim score, aco_diff = compare_ssim(ela_real, aco_real, full=True, gaussian_weights=True)
            train_losses.append(epoch_loss.item())
            mse.append(mse_loss.item())
            bar.update(epoch)
            
            # Logging
            log = {"epoch": epoch, "loss": epoch_loss.item(), "mse (weight %.2f)"%wmse:mse_loss}
            if (epoch % log_frequency == 0 or epoch==nepochs-1):
                print("\n", log)


if __name__ == "__main__":
    # Set seed, clear cache and enable anomaly detection (for debugging)
    set_seed(42)
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)


    # Set training static parameters and hyperparameters
    nepochs=1
    dims_latent = 32                            
    learning_rate=1e-3
    batch_size=4                                        
    device=set_device("cuda", 0)


    # Losses
    l2loss     = nn.MSELoss(reduction="sum")
    awloss     = AWLoss(filter_dim=2, method="fft", reduction="sum", std=3e-4, store_filters="norm", epsilon=3e-15)


    # Dataset
    path = os.path.abspath("/media/dekape/HDD/Ultrasound-MRI-sagittal/")
    train_transform = Compose([
                        Resize(64),
                        Lambda(lambda x: clip_outliers(x)),
                        Lambda(lambda x: scale2range(x, [-1, 1])),
                        ])

    mask = create_mask((64,64), (1,2))

    trainds = MaskedUltrasoundDataset2D(path, 
                                            mode="mri",
                                            transform=train_transform,
                                            mask=mask,
                                            maxsamples=None)
    print(trainds, "\n")
    print(trainds.info(nsamples=1))

    # Dataloader
    train_loader = DataLoader(trainds,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)

    # Sample 
    x_sample, y_sample = trainds[0]


    # Training
    model = make_model(nc=x_sample.shape[0], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train_model(model, optimizer, l2loss, train_loader, wmse=0., valid_loader=None, nepochs=nepochs, log_frequency=1, sample_input=x_sample, sample_target=y_sample, device=device)

    