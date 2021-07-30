###########################
# Utils for using Weights & Biases
############################

import os
import random
import numpy as np
import torch
import progressbar
import pandas as pd
import random
import wandb

from pycm import *

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torch.utils.data import Dataset 

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score

import datasets
import networks
import losses

def setup_config_offline(parameters):
    """
    Sets up wandb.config for pure offline runs, bypassing wanb.init 
    """
    config = wandb.sdk.wandb_config.Config()
    for key in parameters:
        config.__setitem__(key, parameters[key])
    return config

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


def make_dataloader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=2):        
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                        batch_size=batch_size, 
                                        shuffle=shuffle,
                                        pin_memory=pin_memory, 
                                        num_workers=num_workers,
                                        sampler=None)
    return loader


def make_model(config):
    try:
        model = getattr(networks, config.model_name)
    except:
        raise NotImplementedError("Model of name %s has not been found in file networks.py "%config.model_name)
    
    model = model(dims_latent=config.dims_latent)
    model = model.to(config.device)
    config.model = model
    return model


def make_optim(config, model):
    # Optimizer
    try:
        optimizer = getattr(torch.optim, config.optimizer_name)
    except:
        raise NotImplementedError("Optimizer of name %s has not been found in torch.optim"%config.optimizer_name)
    optimizer = optimizer(get_params_to_update(model), lr=config.learning_rate)  
    
    # Momentum
    try:
        for g in optimizer.param_groups:
            g['momentum'] = config.momentum
    except:
        config.momentum = 0.
        pass   

    config.optimizer = optimizer    
    return optimizer


def make_loss(config):
    try:
        criterion = getattr(torch.nn, config.loss_name)
    except:
        try:
            criterion = getattr(losses, config.loss_name)
        except: 
            raise NotImplementedError("Criterion of name %s has not been found in torch.nn or in losses.py"%config.loss_name)

    criterion = criterion()   
    criterion.reduction = config.reduction 
    criterion.store_filters = config.store_filters
    
    # Diagonal stabilisation percent and perturbation (applicable to AWLoss)
    criterion.alpha = config.alpha
    criterion.epsilon = config.epsilon
        
    config.criterion = criterion
    return criterion


def set_up_train(config, train_dataset, valid_dataset=None, train_loader=None, valid_loader=None, model=None, optimizer=None, criterion=None):
    # Update config with train_dataset info
    config.train_dataset = train_dataset
    config.valid_dataset = valid_dataset
    sample = train_dataset[0][0]
    config.input_channels, config.input_width, config.input_height = sample.shape

    # Make data loaders
    if train_loader is None:
        train_loader = make_dataloader(train_dataset, config.batch_size)
    if config.validate and valid_loader is None:
        try:
            valid_loader = make_dataloader(valid_dataset, config.test_batch_size)
        except Exception as e:
            valid_loader = None
            raise e

    # Make model
    if model is None:
        model = make_model(config)
    else:
        config.model_name = str(model)
        
        config.model = model
    
    # Make optimizer
    if optimizer is None:
        optimizer = make_optim(config, model)
    else:
        config.optimizer_name=str(optimizer)
        config.optimizer=optimizer
        config.momentum = optimizer.momentum
           
    # Make loss
    if criterion is None:
        criterion = make_loss(config)
    else:
        config.loss_name=str(criterion)
        config.criterion = criterion
        config.store_filters = criterion.store_filters
        config.alpha = criterion.alpha
        config.epsilon = criterion.epsilon
        config.store_filters = criterion.store_filters
    
    
    #### Print summary ####
    # Config items
    for item in config.items():
        print_single_stats(item[0], item[1])
    
    # Number of train samples by category
    print_single_stats("\nTotal train samples", "    %i"%len(train_dataset))

    # Number of valid samples by category
    if config.validate and valid_dataset is not None:
        print_single_stats("\nTotal valid samples","    %i"%len(valid_dataset))

    # Check model compatibility with input size
    print("\nTesting model compatibility with input size...")
    sample_input = torch.zeros_like(train_dataset[0][0]).unsqueeze(0).expand(config.batch_size, -1, -1, -1).to(config.device)
    print_single_stats("Sample input shape", sample_input.shape)
    sample_output = model(sample_input)
    if isinstance(sample_output, tuple):
        sample_output = sample_output[0]
    print_single_stats("Sample output shape", sample_output.shape)

    return model, criterion, optimizer, train_loader, valid_loader 