import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.datasets import MNIST, CIFAR10

from networks import *
from utils import *
from losses import *

import matplotlib.pyplot as plt

   
if __name__ == "__main__":
    # Set training and testing/validate transforms for normalisation and data augmentation
    train_transform = Compose([
        Resize(28),
        ToTensor(),
        # Normalize(0.1307, 0.3081),
    ])
    

    # Set training static parameters and hyperparameters
    parameters = dict(
        nepochs=2000,
        dims_latent = 10,                              
        learning_rate=1e-3,
        batch_size=1,                                        
        transform=train_transform,
        device=set_device("cuda"),
        )
  
    
    # Model saving settings
    save_model = False
    save_frequency = 5
    
    # access all HPs through wandb.config, so logging matches execution!
    config = setup_config_offline(parameters)

    # Set seed, clear cache
    set_seed(42)
    torch.cuda.empty_cache()
    
    # Data
    trainds = MNIST("./", download=False, train=True, transform=parameters["transform"])
    X = trainds[0][0].unsqueeze(0)
    plt.imsave("orig.png", X[0, 0], cmap="gray")
    
    # Model and optim
    model = Autoencoder(dims_latent=2, nc=X.size(0))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)


    # Input / Output test
    print(X.shape, model(X).shape)

    # Losses: total_loss = a*awi + w*mse +  c*total_variation
    w, mse = 1., torch.nn.MSELoss()
    a, awi = 0., AWILoss2D()
    c, tv =  0., TV()
    switch_at_epoch = 0 # swtich from pure mse to total loss, set to 0 to disable    
    # Send to devices
    X = X.to(config.device)
    model = model.to(config.device)

    # Training loop
    print("\n\nTraining started ...")    
    for epoch in range(config.nepochs):
        model.train()
        optimizer.zero_grad()
        
        recon = model(X)
        mse_loss = mse(recon, X)
        awi_loss, _, _ = awi(recon, X) 
        tv_loss = tv(recon)
        
        if epoch < switch_at_epoch:
            mse_loss.backward()
        else:
            total_loss = a*awi_loss + w*mse_loss + c*tv_loss
            total_loss.backward()
        optimizer.step()
    
        log = {"epoch": epoch, "mse_loss":mse_loss.item(), "awi_loss":awi_loss.item(), "tv_loss": tv_loss.item()}
        if epoch % 100 == 0 or epoch==config.nepochs-1: 
            print(log)
            plt.imsave("recon.png", recon[0, 0].cpu().detach().numpy(), cmap="gray")
            plt.imsave("diff.png", X[0, 0].cpu().detach().numpy() - recon[0, 0].cpu().detach().numpy())
        
        
      
    