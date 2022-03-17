import sys 
sys.path.append('..')

import torch
import torch.nn.functional as F
import numpy as np
from scipy import signal

from networks import *
from utils import *
from losses import *

import matplotlib.pyplot as plt
from matplotlib import cm
from math import ceil


def get_middle_arr(arr, length):
    len_arr = len(arr)
    start = int(len_arr/2) - int(length/2)
    return arr[start:start+length]


def get_middle_arr2d(img, shape):
    x_len, y_len = shape[0], shape[1]
    x_start = int(img.shape[0]/2) - int(x_len/2)
    y_start = int(img.shape[1]/2) - int(y_len/2)
    return img[x_start:x_start+x_len, y_start:y_start+y_len]

    
def circle_img(shape, lag=(0, 0), fill_val=0, circle_val=1., radius=1.):
    im = torch.zeros(shape) + fill_val
    
    xarr = torch.linspace(-shape[0], shape[0], shape[0])
    yarr = torch.linspace(-shape[1], shape[1], shape[1])
    xx, yy = torch.meshgrid(xarr, yarr)
    
    lagy, lagx = lag
    dist = torch.sqrt((xx-lagx)**2 + (yy-lagy)**2)
    idx = torch.where(dist <= radius*2)
    im[idx[0], idx[1]] = circle_val   
    
    return im

def square_img(shape, lag=(0, 0), fill_val=0, square_val=1., radius=1.):
    im = torch.zeros(shape) + fill_val
    
    xarr = torch.linspace(-shape[0], shape[0], shape[0])
    yarr = torch.linspace(-shape[1], shape[1], shape[1])
    
    lagx, lagy = lag
    
    idx = torch.where(xarr > 0 + lagx)[0][0]
    idy = torch.where(yarr > 0 + lagy)[0][0]
    im[idx-radius:idx+radius, idy-radius:idy+radius] = square_val   
    
    return im



def test_filter_img(input, target, awi_loss, figtitle=None, cmap=None, vmin=-1, vmax=1, errvmin=None, errvmax=None):
    """
    input: [batch_size, no_of_channels, height, width]
    target: [batch_size, no_of_channels, height, width]
    awi_loss: instance of a child class of AWLoss
    """
    assert input.shape == target.shape
    assert len(input.shape) == 4
    assert input.shape[0] == 1, "Test only viable for a single sample but found %g"%(input.shape[0])
    
    # Get dimensionalities, squeeze input and output
    input, target = input.squeeze(0), target.squeeze(0)
    nc, h, w = input.shape
    
    # Evaluate loss, retrieve filter and penalty
    # This is a reverse AWI implementation, but here we're looking for the filter that transforms the input to target, so we reverse the parameters in the loss)
    if awi_loss.store_filters is False:
        awi_loss.store_filters = "unorm"

    f = awi_loss(target.unsqueeze(0), input.unsqueeze(0))
    v, T = awi_loss.filters[0], awi_loss.T

    # Filter dimension (1D = 0 or 2D = 2)
    nd = len(v.shape[1:])
    
    # Reconstruct target by convoling input with filter
    if nd == 0:
        D = signal.convolve(input.flatten(start_dim=0), v, mode="full")
        D = get_middle_arr(D, nc*h*w).reshape(nc, h, w)
    elif nd == 2:
        D = np.zeros(input.shape) - 1
        for i in range(nc): 
            D[i]= get_middle_arr2d(signal.convolve2d(input[i], v[i]), (h, w))
    else:
        raise Exception(" Only supporting filters of dimensions 1 and 2, but found %g dimensions"%nd)


    # Transpose for plotting
    input, target, D = input.numpy().transpose(1,2,0), target.numpy().transpose(1,2,0), D.transpose(1,2,0)


    # Plotting input, target, recon and error
    fig, ax = plt.subplots(1, 6, figsize=(30, 4))
    if figtitle is not None:
        fig.suptitle(figtitle, fontsize=16)
        
    norm = cm.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    im = ax[0].imshow(input, norm=norm, cmap=cmap)
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Input")

    im = ax[1].imshow(target, norm=norm, cmap=cmap)
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("target")

    im = ax[2].imshow(D, norm=norm, cmap=cmap)
    fig.colorbar(im, ax=ax[2])
    ax[2].set_title("input*filter")
    
    im = ax[3].imshow(D - target, norm=cm.colors.Normalize(errvmin, errvmax), cmap=cmap)
    fig.colorbar(im, ax=ax[3])
    ax[3].set_title("input*filter - target")


    # Plot filter and penalty functions
    if nd == 0:
        xarr = torch.linspace(-len(v.flatten()), len(v.flatten()),  len(v.flatten())).float()
        
        ax[4].plot(xarr, v.flatten())
        peak = int(xarr[torch.argmax(torch.abs(v.flatten())).item()])
        ax[4].set_title("Filter / len=%.3f/ lag peak=%g / loss=%.3f"%(v.sum(), peak, f))
        ax[4].set_xlabel("Lag")
        # ax[4].set_ylim(None, 1.1)
        
        ax[5].plot(xarr, T.detach())
        peak = int(xarr[torch.argmax(T).item()])
        ax[5].set_title("penalty function, min lag %g "%peak)
        ax[5].set_xlabel("Lag")
        ax[5].set_ylim(-0.1, 1.1)
        
    elif nd == 2:
        xarr = np.linspace(-v[0].shape[0], v[0].shape[0], v[0].shape[0])
        yarr = np.linspace(-v[0].shape[1], v[0].shape[1], v[0].shape[1])
        xx, yy = np.meshgrid(xarr, yarr)
        extent = [-v[0].shape[0], v[0].shape[0], -v[0].shape[1], v[0].shape[1] ]

                    
        im  = ax[4].imshow(v.numpy().transpose(1,2,0), extent=extent)
        fig.colorbar(im, ax=ax[4])
        peaky, peakx = torch.where(torch.abs(v[0]) == torch.max(torch.abs(v[0])))
        peaky, peakx  = int(yarr[peaky.item()]), int(xarr[peakx.item()])
        ax[4].set_title("Filter 2D / len={:.3f}/ lag peak=({:d}, {:d})/ loss={:.3f}".format(v[0].sum(), peaky, peakx, f))
        ax[4].set_xlabel("Lag X")
        ax[4].set_ylabel("Lag Y")
        
        im = ax[5].imshow(T.unsqueeze(0).repeat(nc, 1, 1).permute(1,2,0).detach(), extent=extent)
        peaky, peakx = torch.where(T == T.max())
        peaky, peakx  = int(yarr[peaky]), int(xarr[peakx])
        fig.colorbar(im, ax=ax[5])
        ax[5].set_title("penalty function, min lag  ({:d}, {:d}) ".format(peaky, peakx))
        ax[5].set_xlabel("Lag X")
        ax[5].set_ylabel("Lag Y")
        
    plt.show()



def test_filter_1dsignal(input, target, awi_loss, figtitle=None, ymin=None, ymax=None, errymin=None, errymax=None):
    """
    input: [batch_size, signal_length]
    target: [batch_size, signal_length]
    awi_loss: instance of a child class of AWLoss
    """
    assert input.shape == target.shape
    assert len(input.shape) == 2
    assert input.shape[0] == 1, "Test only viable for a single sample but found %g"%(input.shape[0])
    assert "2D" not in str(awi_loss), "Please use 1D implementations of AWLoss"
    
    # Get dimensionalities
    n = input.shape[-1]
    
    # Evaluate loss, retrieve filter and penalty
    # This is a reverse AWI implementation, but here we're looking for the filter that transforms the input to target, so we reverse the parameters in the loss)
    if awi_loss.store_filters is False:
        awi_loss.store_filters = "unorm"

    f = awi_loss(target, input)
    v, T = awi_loss.filters[0], awi_loss.T

    # Reconstruct target signal from input and filter
    D = signal.convolve(input.flatten(start_dim=0), v, mode="full")
    D = get_middle_arr(D, n)


    # Transpose for plotting
    input, target = input.squeeze(0).numpy(), target.squeeze(0).numpy()


    # Plotting input, target, recon and error
    fig, ax = plt.subplots(1, 6, figsize=(30, 4))
    if figtitle is not None:
        fig.suptitle(figtitle, fontsize=16)
        
    im = ax[0].plot(input)
    ax[0].set_ylim(ymin, ymax)
    ax[0].set_title("Input")

    im = ax[1].plot(target, c="r")
    ax[1].set_ylim(ymin, ymax)
    ax[1].set_title("target")

    im = ax[2].plot(D)
    ax[2].set_ylim(ymin, ymax)
    ax[2].set_title("input*filter")
    
    im = ax[3].plot(D - target, c="green")
    ax[3].set_ylim(errymin, errymax)
    ax[3].set_title("input*filter - target")


    # Plot filter and penalty functions
    xarr = torch.linspace(-len(v.flatten()), len(v.flatten()),  len(v.flatten())).float()
    
    ax[4].plot(xarr, v.flatten())
    peak = int(xarr[torch.argmax(torch.abs(v.flatten())).item()])
    ax[4].set_title("Filter / len=%.3f/ lag peak=%g / loss=%.3f"%(v.sum(), peak, f))
    ax[4].set_xlabel("Lag")
    
    ax[5].plot(xarr, T.detach())
    peak = int(xarr[torch.argmax(T).item()])
    ax[5].set_title("penalty function, min lag %g "%peak)
    ax[5].set_xlabel("Lag")
    ax[5].set_ylim(-0.1, 1.1)
        
    plt.show()
