import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def set_parameter_requires_grad(model, requires_grad=False):
    """https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html"""
    for param in model.parameters():
        param.requires_grad = requires_grad
    return None
            
class Mish(nn.Module):
  def __init__(self):
    super(Mish, self).__init__()
  def forward(self, x):
    return(x * torch.tanh(F.softplus(x)))

class Sine(nn.Module):
  def __init__(self):
    super(Sine, self).__init__()
  def forward(self, x):
    return torch.sin(x)

  
class Autoencoder(nn.Module):
  def __init__(self, dims_latent, nc=1, h=28):
    '''
    Class combines the Encoder and the Decoder with an Autoencoder latent space.

    dims_latent: [int] the dimension of (number of nodes in) the mean-field gaussian latent variable
    '''

    super(Autoencoder, self).__init__()
    self.nc, self.h = nc, h

    # Encoder layers
    self.e_fc0 = nn.Linear(nc*h*h, 512)  # Image to hidden, fully connected
    self.e_fc1 = nn.Linear(512, 256)  # Image to hidden, fully connected
    self.e_fc2 = nn.Linear(256, dims_latent)  # Hidden to latent, fully connected

    # Decoder layers
    self.d_fc0 = nn.Linear(dims_latent, 256)  # Connectivity Latent to Hidden
    self.d_fc1 = nn.Linear(256, 512)
    self.d_fc2 = nn.Linear(512, nc*h*h)
    
    # Activation
    self.activation = Mish()
    # self.activation = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    

  def encoder(self, x):
    z = torch.flatten(x, start_dim = 1)  # Reshape the input into a vector (nD to 1D) 
    z = self.activation(self.e_fc0(z))  # Run Image through Linear transform then activation function
    z = self.activation(self.e_fc1(z))  # Run Image through Linear transform then activation function
    z = self.activation(self.e_fc2(z))  # Run Image through Linear transform then activation function
    return z

  def decoder(self, z):
    x_rec = self.activation(self.d_fc0(z))
    x_rec = self.activation(self.d_fc1(x_rec))
    x_rec = self.sigmoid(self.d_fc2(x_rec))
    x_rec = x_rec.reshape(-1, self.nc, self.h, self.h)
    return x_rec

  def forward(self, x, noise_fac=0):
    z = self.encoder(x)  # Run the image through the Encoder
    x_rec = self.decoder(z + noise_fac*torch.rand_like(z))  
    return x_rec  # Return the output of the decoder (the reconstructed image)


class VAE(nn.Module):
  def __init__(self, dims_latent, nc=1, h=28):
    '''
    Class combines the Encoder and the Decoder with an VAE latent space.

    dims_latent: [int] the dimension of (number of nodes in) the mean-field gaussian latent variable
    '''

    super(VAE, self).__init__()
    self.nc, self.h = nc, h

    # Encoder layers
    self.e_fc0 = nn.Linear(self.nc*self.h*self.h, 512)  # Image to hidden, fully connected
    self.e_fc1 = nn.Linear(512, 256)  # Image to hidden, fully connected
    self.mu = nn.Linear(256, dims_latent)  # Hidden to latent, fully connected mean values of multivariate
    self.sigma = nn.Linear(256, dims_latent)  # Hidden to latent, fully connected variance values of multivariate

    # Decoder layers
    self.d_fc0 = nn.Linear(dims_latent, 256)  # Connectivity Latent to Hidden
    self.d_fc1 = nn.Linear(256, 512)
    self.d_fc2 = nn.Linear(512, self.nc*self.h*self.h)
    
    # Distribution for sampling
    self.distribution = torch.distributions.Normal(0, 1)
    
    # Activation
    self.activation = Mish()
    self.sigmoid = nn.Sigmoid()

    
  def encoder(self, x):
    x = torch.flatten(x, start_dim = 1)  # Reshape the input into a vector (nD to 1D) 
    x = self.activation(self.e_fc0(x))  
    x = self.activation(self.e_fc1(x))  
    mu, sigma = self.mu(x), torch.exp(self.sigma(x)) # Exponential activation ensures positivity for Sigma
    return mu, sigma

  def decoder(self, z):
    x_rec = self.activation(self.d_fc0(z))
    x_rec = self.activation(self.d_fc1(x_rec))
    x_rec = self.sigmoid(self.d_fc2(x_rec)) #sigmoid because want values from 0 to 1
    x_rec = x_rec.reshape(-1, self.nc, self.h, self.h) 
    return x_rec
  
  def sample_latent(self, mu, sigma, device="cpu"):
    z = mu + sigma * self.distribution.sample(mu.shape).to(device)
    return z

  def forward(self, x, noise_fac=0):
    mu, sigma = self.encoder(x)  # Run the image through the Encoder
    z = self.sample_latent(mu, sigma, device=x.device)
    x_rec = self.decoder(z + noise_fac*torch.rand_like(z))  
    return x_rec, mu, sigma  # Return the output of the decoder (the reconstructed image), mu and sigma vals


class CAE28(nn.Module):
    def __init__(self, dims_latent, nc=1):
      super(CAE28, self).__init__()

      # Activation
      self.activation = Mish()
      self.sigmoid = nn.Sigmoid()
      self.sine = Sine()

      # Encoder Layers
      self.e_cv1 = nn.Conv2d(in_channels=nc,  out_channels=32,  kernel_size=4, stride=2, padding=0)
      self.e_cv2 = nn.Conv2d(in_channels=32,  out_channels=64,  kernel_size=4, stride=2, padding=0)
      self.e_cv3 = nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=4, stride=2, padding=0)
      self.e_fc5 = nn.Linear(128, dims_latent)

      # Decoder Layers 
      self.d_fc1 = nn.Linear(dims_latent, 128)
      self.d_cv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=3, padding=0)
      self.d_cv3 = nn.ConvTranspose2d(in_channels=64,  out_channels=32, kernel_size=3, stride=3, padding=0)
      self.d_cv4 = nn.ConvTranspose2d(in_channels=32,  out_channels=nc, kernel_size=4, stride=3, padding=0)

    def encoder(self, x):
      z = self.activation(self.e_cv1(x))
      z = self.activation(self.e_cv2(z))
      z = self.activation(self.e_cv3(z))
      z = self.activation(self.e_fc5(z.flatten(start_dim=1)))
      return z

    def decoder(self, z):
      x = self.activation(self.d_fc1(z))
      x = self.activation(self.d_cv2(x.view(-1, 128, 1, 1) ))
      x = self.activation(self.d_cv3(x))
      x = self.sigmoid(self.d_cv4(x)) 
      return x

 
    def forward(self, x, noise_fac=0):
      z = self.encoder(x)  # Run the image through the Encoder
      x_rec = self.decoder(z + noise_fac*torch.rand_like(z))  
      return x_rec  # Return the output of the decoder (the reconstructed image)
    
class CAE32(nn.Module):
    def __init__(self, dims_latent, nc=1, init_channels=16):
      super(CAE32, self).__init__()

      # Activation
      self.activation = nn.ReLU()
      self.activation = Mish()
      self.sigmoid = nn.Sigmoid()
      
      self.init_channels=init_channels

      # Encoder Layers
      self.e_cv1 = nn.Conv2d(in_channels=nc,  out_channels=self.init_channels,  kernel_size=4, stride=2, padding=1)
      self.e_cv2 = nn.Conv2d(in_channels=self.init_channels,  out_channels=self.init_channels*2,  kernel_size=4, stride=2, padding=1)
      self.e_cv3 = nn.Conv2d(in_channels=self.init_channels*2,  out_channels=self.init_channels*4, kernel_size=4, stride=2, padding=1)
      self.e_cv4 = nn.Conv2d(in_channels=self.init_channels*4,  out_channels=self.init_channels*8, kernel_size=4, stride=2, padding=0)
      self.e_fc5 = nn.Linear(self.init_channels*8, dims_latent)
      
      # Adaptive pool
      self.avg_pool = nn.AdaptiveAvgPool2d(1)

      # Decoder Layers 
      self.d_fc1 = nn.Linear(dims_latent, self.init_channels*8)
      self.d_cv2 = nn.ConvTranspose2d(in_channels=self.init_channels*8, out_channels=self.init_channels*4, kernel_size=4, stride=1, padding=0)
      self.d_cv3 = nn.ConvTranspose2d(in_channels=self.init_channels*4,  out_channels=self.init_channels*2, kernel_size=4, stride=2, padding=1)
      self.d_cv4 = nn.ConvTranspose2d(in_channels=self.init_channels*2,  out_channels=self.init_channels, kernel_size=4, stride=2, padding=1)
      self.d_cv5 = nn.ConvTranspose2d(in_channels=self.init_channels,  out_channels=nc, kernel_size=4, stride=2, padding=1)

    def encoder(self, x):
      z = self.activation(self.e_cv1(x))
      z = self.activation(self.e_cv2(z))
      z = self.activation(self.e_cv3(z))
      z = self.activation(self.e_cv4(z))
      z = self.avg_pool(z).flatten(start_dim = 1)
      z = self.activation(self.e_fc5(z))
      return z

    def decoder(self, z):
      x = self.activation(self.d_fc1(z))
      x = self.activation(self.d_cv2(x.view(-1, self.init_channels*8, 1, 1) ))
      x = self.activation(self.d_cv3(x))
      x = self.activation(self.d_cv4(x))
      x = self.sigmoid(self.d_cv5(x))
      return x

 
    def forward(self, x, noise_fac=0):
      z = self.encoder(x)  # Run the image through the Encoder
      x_rec = self.decoder(z + noise_fac*torch.rand_like(z))  
      return x_rec  # Return the output of the decoder (the reconstructed image)


class CVAE28(nn.Module):
    def __init__(self, dims_latent, nc=1):
      super(CVAE28, self).__init__()

      # Activation
      self.activation = Mish()
      self.sigmoid = nn.Sigmoid()

      # Encoder Layers
      self.e_cv1 = nn.Conv2d(in_channels=nc,  out_channels=32,  kernel_size=4, stride=2, padding=0)
      self.e_cv2 = nn.Conv2d(in_channels=32,  out_channels=64,  kernel_size=4, stride=2, padding=0)
      self.e_cv3 = nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=4, stride=2, padding=0)
      self.e_mu, self.e_sigma = nn.Linear(128, dims_latent), nn.Linear(128, dims_latent)

      # Decoder Layers 
      self.d_fc1 = nn.Linear(dims_latent, 128)
      self.d_cv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=3, padding=0)
      self.d_cv3 = nn.ConvTranspose2d(in_channels=64,  out_channels=32, kernel_size=3, stride=3, padding=0)
      self.d_cv4 = nn.ConvTranspose2d(in_channels=32,  out_channels=nc, kernel_size=4, stride=3, padding=0)

      # Distribution for sampling
      self.distribution = torch.distributions.Normal(0, 1)

    def encoder(self, x):
      z = self.activation(self.e_cv1(x))
      z = self.activation(self.e_cv2(z))
      z = self.activation(self.e_cv3(z))
      mu, sigma = self.e_mu(z.flatten(start_dim=1)), torch.exp(self.e_sigma(z.flatten(start_dim=1)))
      return mu, sigma

    def decoder(self, z):
      x = self.activation(self.d_fc1(z))
      x = self.activation(self.d_cv2(x.view(-1, 128, 1, 1) ))
      x = self.activation(self.d_cv3(x))
      x = self.sigmoid(self.d_cv4(x)) 
      return x

    def sample_latent(self, mu, sigma, device="cpu"):
      z = mu + sigma * self.distribution.sample(mu.shape).to(device)
      return z

    def forward(self, x, noise_fac=0):
      mu, sigma = self.encoder(x)  # Run the image through the Encoder
      z = self.sample_latent(mu, sigma, device=x.device)
      x_rec = self.decoder(z + noise_fac*torch.rand_like(z))  
      return x_rec, mu, sigma  # Return the output of the decoder (the reconstructed image), mu and sigma vals

# if __name__=="__main__":
#   X = torch.zeros(1,1,28,28)
#   model = ConvAE(dims_latent=10)
#   print(model(X).shape)