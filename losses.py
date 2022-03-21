from math import floor
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import abc

class TV(nn.Module):
  def __init__(self):
    super(TV, self).__init__()

  def forward(self, x):
      """Total variation """
      reg = (torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + 
      torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
      return reg

class KLD(nn.Module):
  def __init__(self):
    super(KLD, self).__init__()

  def forward(self, mu, sigma):
      """Kl Divergence """
      return (sigma**2 + mu**2 - torch.log(sigma**2) - 1/2).sum()

class AWLoss(nn.Module):
    def __init__(self, epsilon=0., std=1e-4, reduction="sum", store_filters=False) :
        super(AWLoss, self).__init__()
        self.epsilon = epsilon
        self.std = std
        self.filters = None
        self.T = None
        self.current_epoch = 0

        if store_filters in ["norm", "unorm"] or store_filters is False:
            self.store_filters=store_filters
        else:
            raise ValueError("store_filters must be 'norm', 'unorm' or False, but found {}".format(store_filters))

        if reduction=="mean" or reduction=="sum":
            self.reduction = reduction
        else:
            raise ValueError("reduction must be 'mean' or 'sum', but found {}".format(reduction))

    def update_epoch(self):
        self.epoch = self.current_epoch + 1 

    def make_toeplitz(self, a):
        "Makes toeplitz matrix of a vector A"
        h = a.size(0)
        A = torch.zeros((3*h-2, 2*h-1), device=a.device)
        for i in range(2*h-1):
            A[i:i+h, i] = a[:]  
        A = A.to(a.device)
        return A

    def make_doubly_block(self, X):
        """Makes Doubly Blocked Toeplitz of a matrix X [r, c]"""
        r_block = 3 * X.shape[1] -2                       # each row will have a toeplitz matrix of rowsize 3*X.shape[1] - 2
        c_block = 2*X.shape[1]  -1                        # each row will have a toeplitz matrix of colsize 2*X.shape[1] - 1
        n_blocks = X.shape[0]                             # how many rows / number of blocks
        r = 3*(n_blocks * r_block) -2*r_block             # total number of rows in doubly blocked toeplitz
        c = 2*(n_blocks * c_block) -1*c_block             # total number of cols in doubly blocked toeplitz
        
        Z = torch.zeros(r, c, device=X.device)
        for i in range(X.shape[0]):
            row_toeplitz = self.make_toeplitz(X[i])
            for j in range(2*n_blocks - 1):
                ridx = (i+j)*r_block
                cidx = j*c_block
                Z[ridx:ridx+r_block, cidx:cidx+c_block] = row_toeplitz[:, :]
        return Z  

    def pad_edges_to_len(self, x, length, val=0):
        """
        x must be a 1d signal of shape [batch_size, signal_length]
        """
        total_pad = length - x.shape[1]
        pad_lef = floor(total_pad / 2)
        pad_rig = total_pad - pad_lef
        return nn.ConstantPad1d((pad_lef, pad_rig), val)(x)

    def pad_edges_to_shape(self, x, shape, val=0):
        """
        x must be a multichannel signal of shape [batch_size, nchannels, width, height]
        """
        pad_top, pad_lef = floor((shape[0] - x.shape[2])/2), floor((shape[1] - x.shape[3])/2)
        pad_bot, pad_rig = shape[0] - x.shape[2] - pad_top, shape[1] - x.shape[3] - pad_lef
        return nn.ConstantPad2d((pad_lef, pad_rig, pad_top, pad_bot), val)(x)

    def gaussian(self, xarr, a, std, mean, dim=1):
        return a*torch.exp(-(xarr - mean)**2 / (2*std**2))

    def gauss2d(self, x=0, y=0, mx=0, my=0, sx=1., sy=1., a=1.):
        return a / (2. * np.pi * sx * sy) * torch.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

    def penalty(self, xarr, std=1.):
        tarr = self.gaussian(xarr=xarr, a=1.0, std=std, mean=0)
        tarr = tarr / torch.max(torch.abs(tarr))
        return  tarr

    def penalty2d(self, shape, stdx=1., stdy=1., device="cpu"):
        xarr = torch.linspace(-10., 10., shape[0], requires_grad=True, device=device)
        yarr = torch.linspace(-10., 10., shape[1], requires_grad=True, device=device)
        xx, yy = torch.meshgrid(xarr, yarr)

        # Adjust the location of the zero-lag of the T function to match the location of the expected delta spike
        dispx, dispy = (len(xarr) % 2 - 1) / 2, (len(yarr) % 2 - 1) / 2
        dx, dy = (xarr[-1] - xarr[0]) / (len(xarr) - 1), (yarr[-1] - yarr[0]) / (len(yarr) - 1)

        tarr = self.gauss2d(xx, yy, mx=dx*dispx, my=dy*dispy, sx=stdx, sy=stdy, a=1.0)
        tarr = tarr / torch.max(torch.abs(tarr)) # normalise amplitude of T
        return tarr.to(device)
        
    def norm(self, A, dim=()):
        return torch.sqrt(torch.sum(A**2, dim=dim))

    def forward(self, *args):
        raise NotImplementedError ("Please use derived classes to implement AWLoss")

class AWLoss1D(AWLoss):
    def __init__(self, *args, **kwargs) :
        super(AWLoss1D, self).__init__(*args, **kwargs)
    
    def forward(self, recon, target):
        """
        Adaptive Weiner Loss Computation
        Loss is based on reverse AWI, which makes use of a more efficient computational graph (see paper)
        In reverse FWI, the filter v computed below is the filter that transforms  "target" into "recon"

        """
        assert recon.shape == target.shape
        recon, target = recon.flatten(start_dim=1), target.flatten(start_dim=1)
        # recon, target = torch.flip(recon, dims=(0,1)), torch.flip(target, dims=(0,1))

        
        self.T = self.penalty(torch.linspace(-1., 1., 2*recon.shape[1]-1, requires_grad=True), self.std).to(recon.device)
        if self.store_filters: self.filters = torch.zeros(recon.shape[0], 2*recon.shape[1]-1 ).to(recon.device) if self.store_filters else None
        
        f = 0.
        for i in range(recon.size(0)):
            # Compute filter
            D = self.make_toeplitz(target[i])
            D_t = D.T
            v = D.T @ D
            v = v + torch.diag(torch.zeros_like(torch.diagonal(v)) + torch.abs(v).max()*self.epsilon)
            v = torch.inverse(v)
            v = v @ (D_t @ self.pad_edges_to_len(recon[i].unsqueeze(0), D_t.shape[1])[0])
            
            # Normalise filter and store if prompted
            if self.store_filters=="unorm": self.filters[i] = v[:]
            v = v / self.norm(v)
            if self.store_filters=="norm": self.filters[i] = v[:]

            # Compute functional
            f = f + 0.5 * self.norm(self.T - v) #+ 100*self.norm(v)
                
        if self.reduction == "mean":
            f = f / recon.size(0)

        return f
      
      
class AWLoss2D(AWLoss):
    def __init__(self, *args, **kwargs) :
        super(AWLoss2D, self).__init__(*args, **kwargs)
      
    
    def forward(self, recon, target):
        """
        g = || P*W - D ||^2
        g = || Zw - d ||^2
        dgdw = Z^T (Zw - d)
        dgdw --> 0 : w = (Z^T @ Z)^(-1) @ Z^T @ d 

        To stabilise the matrix inversion, an amount is added to the diagonal of (Z^T @ Z)
        based on alpha and epsilon values such that the inverted matrix is 
        (Z^T @ Z) + alpha*diagonal(Z^T @ Z) + epsilon
        
        Working to minimise || P*W - D ||^2 where P is the reconstructed image convolved with a 2D kernel W, 
        D is the original/target image, and optimisation aims to force W to be an identity kernel. 
        (https://en.wikipedia.org/wiki/Kernel_(image_processing))
        
        P, D and W here in the code assumed to be single channel and numerically take form of a 2D matrix.
        
        Convolving P with W (or W with P) is equivalent to the matrix vector multiplication Zd where Z is the 
        doubly block toeplitz of the reconstructed image P and w is the flattened array of the 2D kernel W. 
        (https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication)
        
        Therefore, the system is equivalent to solving || Zw - d ||^2, and the solution to w is given by
        w = (Z^T @ Z)^(-1) @ Z^T @ d 
        
        Finally, the function T is an inverse multivariate gaussian in a 2D space to reward when the kernel W is close
        to the identity kernel, and penalise otherwise.
        The value std controls the standard deviation (the spread) of T in both directions and the value a its amplitude
      
        This function applies the reverse AWI formulation (see paper)
        
        """
        assert target.shape == recon.shape
        bs, nc = recon.size(0), recon.size(1)

        filter_shape = (2*recon.shape[2] - 1, 2*recon.shape[3] - 1)
        self.T = self.penalty2d(shape=filter_shape, stdx=self.std, stdy=self.std, device=recon.device)
        if self.store_filters: self.filters = torch.zeros(bs, nc, filter_shape[0], filter_shape[1]).to(recon.device) # one filter per image 

        ## COULD BE VECTORISED? This loop treats every image in batch and every channel of each image as a "separate" sample
        f = 0.
        for i in range(bs): 
          for j in range(nc):
            # Compute filter
            Z = self.make_doubly_block(target[i][j])
            Z_t = Z.T        
            v = Z_t @ Z
            v = v + torch.diag(torch.zeros_like(torch.diagonal(v)) + torch.abs(v).max()*self.epsilon)# stabilise diagonals for matrix inversion
            v = torch.inverse(v) ## COULD BE OPTIMISED?
            v = v @ (Z_t @ self.pad_edges_to_shape(recon[i][j].unsqueeze(0).unsqueeze(0), (3*recon.shape[2] - 2, 3*recon.shape[3] - 2)).flatten(start_dim=0))
            
            # Normalise filter and store if prompted
            if self.store_filters=="unorm": self.filters[i][j] = v[:].view(filter_shape) 
            v = v / self.norm(v)
            if self.store_filters=="norm": self.filters[i][j] = v[:].view(filter_shape) 
            
            # Compute functional
            f = f + 0.5 * self.norm(self.T.flatten() - v) #/ self.norm(v)
            
        if self.reduction=="mean":
          f = f / (bs * nc)
          
        return f


class AWLoss1DFFT(AWLoss):
    def __init__(self, filter_scale=2, *args, **kwargs) :
        super(AWLoss1DFFT, self).__init__(*args, **kwargs)
        self.filter_scale = filter_scale
        
    def wienerfft(self, x, y, prwh=3e-15):
        """
        George Strong (geowstrong@gmail.com)
        calculates the optimal least squares convolutional Wiener filter that 
        transforms signal x into signal y
        """

        assert x.shape == y.shape, "signals x and y must be the same size but are {} and {}".format(x.shape, y.shape)
        
        Fccorr = torch.fft.fft(torch.flip(x, (0,1)), dim=1)*torch.fft.fft(y, dim=1) # cross-correlation of x with y 
        Facorr = torch.fft.fft(torch.flip(x, (0,1)), dim=1)*torch.fft.fft(x, dim=1) # auto-correlation of x
        Fdconv = Fccorr/(Facorr+torch.abs(Facorr).max()*prwh) # deconvolution of Fccorr by Facorr
        rolled = torch.fft.irfft(Fdconv, x.shape[1], dim=1) # inverse Fourier transform
        return torch.roll(rolled, int(-x.shape[1]/2-1), dims=1)

    def forward(self, recon, target, epsilon=None):
        """
        Implements AWLoss using 1D filters and flattened images in the frequency domain
        This function applies the reverse AWI formulatio (see paper)
        """
        assert recon.shape == target.shape
        
        # Flatten recon and target for 1D processing
        recon, target = recon.flatten(start_dim=1), target.flatten(start_dim=1)

        # Define size of the filter, reserve memory to store them if prompted
        filter_size = self.filter_scale*recon.shape[1]
        if filter_size % 2 == 0:
            filter_size = filter_size - 1
        self.filters = torch.zeros(recon.shape[0], filter_size).to(recon.device) if self.store_filters else None

        # Apply padding for FFT convolution
        recon = self.pad_edges_to_len(recon, filter_size)
        target = self.pad_edges_to_len(target, filter_size)

        # Compute weiner filter
        epsilon = self.epsilon if epsilon is None else epsilon
        # v = self.wienerfft(recon, target, epsilon) # forward AWI
        v = self.wienerfft(target, recon, epsilon) # reverse AWI

        # Normalise filter and store if prompted
        if self.store_filters=="unorm": self.filters = v[:]
        v = v / self.norm(v, dim=1).unsqueeze(-1).expand_as(v)
        if self.store_filters=="norm": self.filters = v[:]

        # Penalty function
        self.T = self.penalty(torch.linspace(-1., 1., filter_size, requires_grad=True), self.std).to(recon.device)
        T = self.T.repeat(recon.size(0), 1)

        # Compute loss
        f = 0.5 * self.norm(T - v, dim=1)
        f = f.sum()  
        if self.reduction == "mean":
            f = f / recon.size(0)

        return f


class AWLoss2DFFT(AWLoss):
    def __init__(self, filter_scale=2, *args, **kwargs) :
        super(AWLoss2DFFT, self).__init__(*args, **kwargs)
        self.filter_scale = filter_scale

    def wienerfft2D(self, x, y, prwh=1e-9):
        """
        George Strong (geowstrong@gmail.com)
        calculates the optimal least squares 2D convolutional Wiener filter that 
        transforms 2D signal x into 2D signal y
        """
        assert x.shape == y.shape, "signals x and y must be the same shape"
        Fccorr = torch.fft.fftn(torch.flip(x, (2,3)), dim=(2,3))*torch.fft.fftn(y, dim=(2,3)) # cross-correlation of x with y 
        Facorr = torch.fft.fftn(torch.flip(x, (2,3)), dim=(2,3))*torch.fft.fftn(x, dim=(2,3)) # auto-correlation of x
        Fdconv = Fccorr/(Facorr+torch.abs(Facorr).max()*prwh) # deconvolution of Fccorr by Facorr
        rolled = torch.fft.irfftn(Fdconv, x.shape[2:], dim=(2,3)) # inverse Fourier transform
        return torch.roll(rolled, (int(-x.shape[2]/2-1), int(-x.shape[3]/2-1)), dims= (2,3)) 


    def forward(self, recon, target, epsilon=None):
        """
        Implements AWLoss using 2D filters and multichannel images in the frequency domain
        This function implements the reverse AWI formulation (see paper)
        """
        assert recon.shape == target.shape
        bs, nc = recon.size(0), recon.size(1)


        # Define size of the filter, reserve memory to store them if prompted
        filter_shape = [self.filter_scale*recon.shape[2], self.filter_scale*recon.shape[3] - 1]
        if filter_shape[0] % 2 == 0:
            filter_shape[0] = filter_shape[0] - 1
        if filter_shape[1] % 2 == 0:
            filter_shape[1] = filter_shape[1] - 1
        self.filters = torch.zeros(bs, nc, filter_shape[0], filter_shape[1]).to(recon.device) if self.store_filters else None
        

        # Apply padding for FFT convolution
        recon = self.pad_edges_to_shape(recon, filter_shape)
        target = self.pad_edges_to_shape(target, filter_shape)

        # Compute weiner filter for each channel of each sample in batch
        epsilon = self.epsilon if epsilon is None else epsilon
        v = self.wienerfft2D(target, recon, epsilon) # reverse AWI filter

        # Normalise filter and store if prompted
        if self.store_filters=="unorm": self.filters = v[:]   
        v = v / self.norm(v, dim=(-2,-1)).unsqueeze(-1).unsqueeze(-1).expand_as(v)
        if self.store_filters=="norm": self.filters = v[:]     

        # Penalty function
        self.T = self.penalty2d(shape=filter_shape, stdx=self.std, stdy=self.std, device=recon.device)
        T = self.T.repeat(recon.size(0), recon.size(1), 1 ,1)

        # Compute loss
        f = 0.5 * self.norm(T - v, dim=(-2,-1))
        f = f.sum()
        if self.reduction == "mean":
            f = f / recon.size(0)
        return f


