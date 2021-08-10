import sys 
sys.path.append('./')

import cProfile
from pstats import Stats, SortKey

import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Grayscale
from torchvision.datasets import MNIST, CIFAR10 
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit

from networks import *
from utils import *
from losses import *
from landscape import *
from datasets import *

import matplotlib.pyplot as plt



def run_script(criterion):
    # Set training static parameters and hyperparameters
    dims_latent = 32                            
    learning_rate=1e-3
    batch_size=32                                        
    device=set_device("cuda", 0)
    train_size=0.00108
    
    # Datasets and dataloaders
    train_transform = Compose([
        Resize(28),
        ToTensor(),
    ])

    ds = MNIST("./notebooks/mnist_example", download=False, train=True, transform=train_transform)
    shuffler = StratifiedShuffleSplit(n_splits=1, test_size=1-train_size, random_state=42).split(ds.data, ds.targets)
    train_idx, valid_idx = [(train_idx, validation_idx) for train_idx, validation_idx in shuffler][0]

    X_train, y_train = ds.data[train_idx] / 255., ds.targets[train_idx]
    trainds = TensorDataset(X_train.unsqueeze(1).float(), y_train.float())

    train_loader = DataLoader(trainds, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Sample check
    X = trainds[0][0].unsqueeze(0).to(device)
    print("Trainable Images:", len(trainds))
    print(X.shape, X.min().item(), X.max().item())

    # Model and optimizer
    model = Autoencoder(dims_latent=dims_latent, nc=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    
    #Train one epoch
    _ = train(model, train_loader, optimizer, criterion, device=device)
    


if __name__ == '__main__':
    criterion = AWLoss1DRoll(reduction="sum", std=1e-4, store_filters=True, alpha=0.02)
    filename = "./profile_"+str(criterion)+".txt"
    
    profiler = cProfile.Profile()
    profiler.enable()
    run_script(criterion)
    profiler.disable()
    stats = Stats(profiler).sort_stats('cumtime')
    # stats.strip_dirs()
    stats.print_stats()
    
    # cProfile.run('run_script(criterion)', filename)