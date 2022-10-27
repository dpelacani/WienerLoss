from math import floor
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import abc
from scipy.stats import multivariate_normal

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
    def __init__(self, epsilon=0., gamma=0., eta=0.,  std=1e-4, reduction="sum",
                 method="fft", filter_dim=2, filter_scale=2, store_filters=False,
                 mode="reverse", penalty_function=None) :
        super(AWLoss, self).__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.eta = eta
        self.std = std
        self.filter_scale = filter_scale     
        self.penalty_function = penalty_function
        self.mode = mode

        # Check arguments
        if store_filters in ["norm", "unorm"] or store_filters is False:
            self.store_filters=store_filters
        else:
            raise ValueError("store_filters must be 'norm', 'unorm' or False, but found {}".format(store_filters))

        if reduction=="mean" or reduction=="sum":
            self.reduction = reduction
        else:
            raise ValueError("reduction must be 'mean' or 'sum', but found {}".format(reduction))

        if filter_dim in [1, 2, 3]:
            self.filter_dim = filter_dim
            self.dims = tuple([-i for i in range(self.filter_dim, 0, -1)])
        else:
            raise ValueError("Filter dimenions must be 1, 2 or 3, but found {}".format(filter_dim))

        if method=="fft" or method=="direct":
            self.method = method
            if method=="direct":
                self.filter_scale = 2 # Larger filter scales not supported for direct methods
                if self.filter_dim == 3:
                    raise NotImplementedError("3D filter implementation not available for the direct method")
        else:
            raise ValueError("method must be 'fft' or 'direct', but found {}".format(method))

        # Variables to store metadata
        self.filters = None
        self.T  = None
        self.current_epoch = 0

    def update_epoch(self):
        self.epoch = self.current_epoch + 1

    def norm(self, A, dim=()):
        return torch.sqrt(torch.sum(A**2, dim=dim))

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

    def get_filter_shape(self, input_shape):
        if self.filter_dim == 1:
            _, n = input_shape
            fs = [self.filter_scale*n]
        elif self.filter_dim == 2:
            _, nc, h, w = input_shape
            fs = [nc, self.filter_scale*h, self.filter_scale*w]
        elif self.filter_dim == 3:
            _, nc, h, w = input_shape
            fs = [self.filter_scale*nc, self.filter_scale*h, self.filter_scale*w]

        # Make filter dimensions odd integers
        for i in range(len(fs)): 
            fs[i] = int(fs[i])
            if fs[i] % 2 == 0:
                if (self.filter_dim == 2 and i == 0):  # except nchannels for 2D filters, dimension must match to input
                    pass
                else:
                    fs[i] = fs[i] - 1
        return fs

    def pad_signal(self, x, shape, val=0):
        """
        x must be a multichannel signal of shape [batch_size, nchannels, width, height]
        """
        assert len(x.shape[1:]) == len(shape), "{} {}".format(x.shape, shape)
        pad = []
        for i in range(len(x.shape[1:])):
            p1 = floor((shape[i] - x.shape[i+1])/2)
            p2 = shape[i]-x.shape[i+1] - p1
            pad.extend((p1, p2))
        try:
            pad = [pad[i] for i in [2, 3, 4, 5, 0, 1]]  # permutation of list to agree with nn.functional.pad
        except:
            pass
        return nn.functional.pad(x, tuple(pad), value=val)

    def multigauss(self, mesh, mean, covmatrix):
        """
        Multivariate gaussian of N dimensions on evenly spaced hypercubed grid 
        Mesh should be stacked along the last axis
        E.g. for a 3D gaussian of 20 grid points in each axis mesh should be of shape (20, 20, 20, 3)
        """
        assert len(covmatrix.shape) == 2
        assert covmatrix.shape[0] == covmatrix.shape[1]
        assert covmatrix.shape[0] == len(mean)
        assert len(mesh.shape) == len(mean) + 1, "{} {}".format(len(mesh.shape), len(mean))
        rv = torch.distributions.multivariate_normal.MultivariateNormal(mean, covmatrix)
        rv = torch.exp(rv.log_prob(mesh))
        rv = rv / torch.abs(rv).max()  
        rv = -rv + rv.max() 
        return rv

    def make_penalty(self, shape, std=1e-2, eta=0., penalty_function=None, flip=False,  device="cpu"):
        arr = [torch.linspace(-1., 1., n, requires_grad=True).to(device) for n in shape]
        mesh = torch.meshgrid(arr)
        mesh = torch.stack(mesh, axis=-1)
        if penalty_function is None:
            mean = torch.tensor([0. for i in range(mesh.shape[-1])]).to(device)
            covmatrix = torch.diag(torch.tensor([std**2 for i in range(mesh.shape[-1])])).to(device)
            penalty = self.multigauss(mesh, mean, covmatrix)
            penalty = -penalty + penalty.max() if flip else penalty
        else:
            penalty = penalty_function(mesh)
        penalty = penalty + eta*torch.rand_like(penalty)
        return penalty

    def wienerfft(self, x, y, fs, prwh=1e-9):
        """
        George Strong (geowstrong@gmail.com)
        calculates the optimal least squares convolutional Wiener filter that 
        transforms signal x into signal y using FFT
        """
        assert x.shape == y.shape, "signals x and y must be the same shape"
        Fccorr = torch.fft.fftn(torch.flip(x, self.dims), dim=self.dims)*torch.fft.fftn(y, dim=self.dims) # cross-correlation of x with y 
        Facorr = torch.fft.fftn(torch.flip(x, self.dims), dim=self.dims)*torch.fft.fftn(x, dim=self.dims) # auto-correlation of x
        Fdconv = (Fccorr + prwh) / (Facorr + prwh)  # deconvolution of Fccorr by Facorr
        rolled = torch.fft.irfftn(Fdconv, fs[-self.filter_dim:], dim=self.dims) # inverse Fourier transform
        rolling = tuple([int(-x.shape[i]/2) - 1 for i in range(1, len(x.shape), 1)])[-len(self.dims):]
        return torch.roll(rolled, rolling, dims=self.dims) 

    def wiener(self, x, y, fs, epsilon=1e-9):
        """
        calculates the optimal least squares convolutional Wiener filter that 
        transforms signal x into signal y using the direct Toeplitz matrix implementation
        """
        assert x.shape == y.shape, "signals x and y must be the same shape"
        
        bs = x.shape[0]
        v = torch.empty([bs]+fs, device=x.device)

        if self.filter_dim == 1:
            for i in range(v.shape[0]):
                # Compute filter
                D = self.make_toeplitz(x[i])
                D_t = D.T
                tmp = D.T @ D
                tmp = tmp + torch.diag(torch.zeros_like(torch.diagonal(tmp)) + torch.abs(tmp).max()*epsilon)
                tmp = torch.inverse(tmp)
                v[i] = tmp @ (D_t @ self.pad_signal(y[i].unsqueeze(0), [D_t.shape[1]])[0])

        elif self.filter_dim == 2:
            for i in range(bs): 
                for j in range(x.shape[1]):
                    # Compute filter
                    Z = self.make_doubly_block(x[i][j])
                    Z_t = Z.T        
                    tmp = Z_t @ Z
                    tmp = tmp + torch.diag(torch.zeros_like(torch.diagonal(tmp)) + torch.abs(tmp).max()*self.epsilon)# stabilise diagonals for matrix inversion
                    tmp = torch.inverse(tmp) 
                    tmp = tmp @ (Z_t @ self.pad_signal(y[i][j].unsqueeze(0), (3*y.shape[2] - 2, 3*y.shape[3] - 2)).flatten(start_dim=0))
                    v[i][j] = tmp.reshape(fs[-self.filter_dim:])
        return v


    def forward(self, recon, target, epsilon=None):
        """
        Adaptive Wiener Loss Computation
        Loss is based on reverse AWI, which makes use of a more efficient computational graph (see paper)
        In reverse FWI, the filter v computed below is the filter that transforms  "target" into "recon"

        g = || P*W - D ||^2
        g = || Zw - d ||^2
        dgdw = Z^T (Zw - d)
        dgdw --> 0 : w = (Z^T @ Z)^(-1) @ Z^T @ d 

        To stabilise the matrix inversion, an amount is added to the diagonal of (Z^T @ Z)
        based on alpha and epsilon values such that the inverted matrix is 
        (Z^T @ Z) + max(diagonal(Z^T @ Z)) * epsilon
        
        Working to minimise || P*W - D ||^2 where P is the reconstructed image convolved with a 2D kernel W, 
        D is the original/target image, and optimisation aims to force W to be an identity kernel. 
        (https://en.wikipedia.org/wiki/Kernel_(image_processing))
        
        
        Convolving P with W (or W with P) is equivalent to the matrix vector multiplication Zd where Z is the 
        doubly block toeplitz of the reconstructed image P and w is the flattened array of the 2D kernel W. 
        (https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication)
        
        Therefore, the system is equivalent to solving || Zw - d ||^2, and the solution to w is given by
        w = (Z^T @ Z)^(-1) @ Z^T @ d 
        
        This composes the direct method.

        Alternatively, convolution can be performed in the frequency domain with multiplication and division operations.
        This tends to be much more computationaly efficient.

        Finally, the function T is an inverse multivariate gaussian in a n-dimensional space to reward when the kernel W is close
        to the identity kernel, and penalise otherwise.
        The value std controls the standard deviation (the spread) of T in all directions.
      
        """

        assert recon.shape == target.shape, "recon and target must be of the same shape but found {} and {}".format(recon.shape, target.shape)

        # White noise to recon and target for stabilisation
        recon = recon + self.gamma * torch.rand_like(recon)
        target = target + self.gamma * torch.rand_like(target)

        # Batch size
        bs = recon.shape[0]

        # Flatten recon and target for 1D filters
        if self.filter_dim == 1:
            recon, target = recon.flatten(start_dim=1), target.flatten(start_dim=1)

        # Define size of the filter, reserve memory to store them if prompted
        fs = self.get_filter_shape(recon.shape)
        self.filters = torch.zeros([bs]+fs).to(recon.device) if self.store_filters else None

        # Compute wiener filter
        epsilon = self.epsilon if epsilon is None else epsilon
        if self.method == "fft":
            recon, target = self.pad_signal(recon, fs), self.pad_signal(target, fs)
            if self.mode == "reverse":
                v = self.wienerfft(target, recon, fs, epsilon) # reverse AWI filter
                # print(v.shape)
            elif self.mode == "forward":
                v = self.wienerfft(recon, target, fs, epsilon) # forward AWI filter

        elif self.method == "direct":
            if self.mode == "reverse":
                v = self.wiener(target, recon, fs, epsilon)    # reverse AWI filter
            if self.mode == "forward":
                v = self.wiener(recon, target, fs, epsilon)    # forward AWI filter

        # Normalise filter and store if prompted
        if self.store_filters == "unorm": self.filters = v[:]
        vnorm = torch.norm(v, p=2, dim=self.dims)
        for i in range(self.filter_dim):
            vnorm = vnorm.unsqueeze(-1)
        vnorm = vnorm.expand_as(v)
        if self.store_filters == "norm": self.filters = v[:] / vnorm    

        # Penalty function
        self.T = self.make_penalty(shape=fs[-self.filter_dim:], std=self.std,
                                   eta=self.eta, penalty_function=self.penalty_function,
                                   device=recon.device, flip=True)
        # self.T = torch.clamp(self.T + 0.1, 0, 1)
        T = self.T.unsqueeze(0).expand_as(v)

        # Delta
        self.delta = self.make_penalty(fs[-self.filter_dim:], std=3e-8,
                                       penalty_function=None,
                                       device=recon.device, flip=True)
        delta = self.delta.unsqueeze(0).expand_as(v)

        # Compute Loss
        # f = 0.5 * torch.norm(v * T, p=2, dim=self.dims) / torch.norm(v, p=2, dim=self.dims)
        f = 0.5 * torch.norm(T * (v - delta), p=2, dim=self.dims)

        f = f.sum()
        if self.reduction == "mean":
            f = f / recon.size(0)
        return f


class PixelAWLoss(nn.Module):
    def __init__(self, reduction="sum", epsilon=3e-5):
        super(PixelAWLoss, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, recon, target):
        r = recon/(target + self.epsilon)
        f = torch.abs(torch.ones_like(r) - r)
        f = f.sum()
        if self.reduction == "mean":
            f = f / recon.size(0)

        return f
