import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Grayscale
from torchvision.datasets import MNIST, CIFAR10 
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit

from networks import *
from wandbutils import *
from utils import *
from losses import *
from landscape import *
from datasets import *

import matplotlib.pyplot as plt
import progressbar
import random

if __name__=="__main__":
    # WANDB
    RUN_ONLINE=True
    RUN_NAME=None
    
    # Model and plot saving settings
    save_model = False
    save_plot  = True
    save_frequency = 20

    # Set training static parameters and hyperparameters
    parameters = dict(
        data_name="MNIST",
        nepochs=100,
        batch_size=32,                                        
        device=set_device("cuda"),
        train_split=0.00108,
        validate=False,
        
        # Optimiser
        optimizer_name="Adam",                       
        learning_rate=1e-3,
        
        # Model
        model_name="Autoencoder",
        dims_latent = 32,

        # Loss
        loss_name = "AWLoss1D",
        reduction = "sum",
        store_filters=True,
        alpha=0.02,
        epsilon=0.,
        
        train_transform = Compose([
            Resize(28),
            # ToTensor(),
        ]),
    )
    
    # Set seed, clear cache and enable anomaly detection (for debugging)
    set_seed(42)
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)
    
    # Create dataset with transforms
    ds = MNIST("./notebooks/mnist_example/", download=False, train=True)
    shuffler = StratifiedShuffleSplit(n_splits=1, test_size=1-parameters["train_split"], random_state=42).split(ds.data, ds.targets)
    train_idx, valid_idx = [(train_idx, validation_idx) for train_idx, validation_idx in shuffler][0]

    X_train, y_train = ds.data[train_idx] / 255., ds.targets[train_idx]
    trainds = TransformTensorDataset(X_train.unsqueeze(1).float(), y_train.float(), transform=parameters["train_transform"])
    
    
    
    
    
    
    ###################################################################################################################################
    #                                                  DO NOT CHANGE FROM HERE ON                                                     #
    ###################################################################################################################################

    # Tell wandb to get started
    if RUN_ONLINE:
        wb = wandb.init(project="awloss", config=parameters, entity="dekape", name=RUN_NAME) 
        config = wandb.config
    else:
        config = setup_config_offline(parameters)
    
    # Set up run with parameters
    model, criterion, optimizer, train_loader, valid_loader = set_up_train(config, trainds)
    print(vars(criterion))

    # Let wandb watch the model and the criterion
    if RUN_ONLINE: wandb.watch(model, criterion)
    
    # Training loop
    print("\n\nTraining started ...")
    for epoch in range(config.nepochs):
        epoch_loss, epoch_kl = train(model, train_loader, optimizer, criterion, device=config.device)
        epoch_mse = validate(model, train_loader, nn.MSELoss(reduction="mean"), device=config.device)
        
        log = {"epoch": epoch, "train_loss": epoch_loss.item(), "train_kl": epoch_kl.item(), "train_mse":epoch_mse.item()}
        if RUN_ONLINE: wandb.log(log)
        print(log)

        # Saving model and log plots
        if epoch % save_frequency == 0 or epoch==config.nepochs-1:
            if save_model:
                model_dir =  os.path.join(wb.dir, "%s_epoch%g.pth"%(wb.name, epoch))
                print("\n Saving model at %s \n"%(model_dir))
                torch.save(model.state_dict(), model_dir)
                
            if save_plot and RUN_ONLINE:
                idx = random.randint(0, len(trainds))
                idx=-1
                X = trainds[idx][0].unsqueeze(0).to(config.device)
                recon = model(X)
                if isinstance(recon, tuple): recon = recon[0] # in case model returns more than one output, recon must be the first
                
                f = criterion(recon, X)
                try:
                    v, T = criterion.v_all[0], criterion.T_arr
                except Exception as e:
                    pass

                fig, axs = plt.subplots(2,3)
                plt.suptitle("{} epoch {}".format(criterion, epoch))
                axs[0, 0].imshow(recon[0, 0].cpu().detach().numpy(), cmap="gray", vmin=0., vmax=1.)
                axs[0, 0].set_title("recon")

                axs[0, 1].imshow(X[0, 0].cpu().detach().numpy(), cmap="gray", vmin=0., vmax=1.)
                axs[0, 1].set_title("orig")

                axs[0, 2].imshow(X[0, 0].cpu().detach().numpy() - recon[0, 0].cpu().detach().numpy(), vmin=-0.5, vmax=0.5)
                axs[0, 2].set_title("diff")

                try:
                    axs[1, 0].plot(v.flatten().detach().cpu().numpy())
                    axs[1, 0].plot(T.flatten().detach().cpu().numpy(), "--")
                    axs[1, 0].set_title("T vs v")
                    axs[1, 0].set_ylim(None, 1.1)
                    axs[1, 0].set_title(" argidx T, v: {}, {}".format(torch.argmax(torch.abs(T)).item(), torch.argmax(torch.abs(v)).item()))

                    axs[1, 1].plot((T.flatten() - v.flatten()).detach().cpu().numpy())
                    axs[1, 1].set_title("T - v")
                    axs[1, 1].set_ylim(None, 1.1)
                except:
                    pass
                wandb.log({"recon":plt})
    

    