#!/usr/bin/env python3

import sys, os; sys.path.append(os.path.dirname("./"))
import timeit
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import platform
import cpuinfo

from losses import *
from utils import set_device, set_seed


def plot_convergence(times, n_all, labels, xlabel, ylabel, figtitle=None):
    assert len(times) == len(labels)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if figtitle is not None:
        fig.suptitle(figtitle, fontsize=16)
    
    for i in range(len(times)):
        ax.loglog(n_all, times[i], label=labels[i])
    ax.grid()
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    return None


if __name__ == "__main__":

    # Set seed and device                     
    device=set_device("cuda", 0)
    set_seed(42)
    
    # Losses to evaluate
    losses = [nn.MSELoss(reduction="sum"),
              AWLoss1D(reduction="sum"),
              AWLoss1DFFT(reduction="sum"),
              AWLoss1D(reduction="sum"),
              AWLoss2DFFT(reduction="sum")]
    names = [str(l) for l in losses]

    # Nested list to store performance times
    times = [[] for i in losses ]

    # Profiling for different image sizes
    n_all = [2**i for i in range(1, 6)]
    for n in n_all:
        print(" Processing image size %g"%n)

        # Data for profiling
        X = torch.randn([1, 3, n, n]).to(device)
        target = torch.randn_like(X).to(device)
        
        # Profiling losses
        for i, loss in enumerate(losses):
            times[i].append(timeit.timeit(lambda:loss(X, target), number=30))

    # Environment name
    if device=="cuda":
        envname = torch.cuda.get_device_name(0)
    else:
        try:
            envname = str(cpuinfo.get_cpu_info()['brand_raw'])
        except:
            try:
                envname = str(cpuinfo.get_cpu_info()['brand'])
            except:
                envname = ""
        envname = envname + " " + str(platform.platform())

    # Convergence Plot
    figtitle = "{}".format(envname + "\n Image size $3 x n x n$")
    plot_convergence(times, n_all, names, xlabel=r"$n$", ylabel="Time (s)", figtitle=figtitle)

