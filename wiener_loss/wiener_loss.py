# TODO:
# LDR with pytorch conv
# Sigmoid in penalty
# MSWienerLoss

import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from math import floor


def rms(x):
    square = torch.pow(x, 2)
    mean_square = torch.mean(square)#, dim=self.dims)
    rms = torch.sqrt(mean_square)
    return rms.item()


def pad_signal(x, shape, val=0):
    """
    x must be a multichannel signal of shape
    [batch_size, nchannels, width, height]
    """
    assert len(x.shape[1:]) == len(shape), "{} {}".format(x.shape, shape)
    pad = []
    for i in range(len(x.shape[1:])):
        p1 = floor((shape[i] - x.shape[i+1])/2)
        p2 = shape[i]-x.shape[i+1] - p1
        pad.extend((p1, p2))
    try:
        # permutation of list to agree with nn.functional.pad
        pad = [pad[i] for i in [2, 3, 4, 5, 0, 1]]
    except:
        pass
    return nn.functional.pad(x, tuple(pad), value=val)


def multigauss(mesh, mean, covmatrix):
    """
    Multivariate gaussian of N dimensions on evenly spaced
    hypercubed grid. Mesh should be stacked along the last axis
    E.g. for a 3D gaussian of 20 grid points in each axis mesh
    should be of shape (20, 20, 20, 3)
    """
    assert len(covmatrix.shape) == 2
    assert covmatrix.shape[0] == covmatrix.shape[1]
    assert covmatrix.shape[0] == len(mean)
    assert len(mesh.shape) == len(mean) + 1, \
            "{} {}".format(len(mesh.shape), len(mean))
            
    rv = MultivariateNormal(mean, covmatrix)
    rv = torch.exp(rv.log_prob(mesh))
    rv = rv / torch.abs(rv).max()
    return rv


def identity(mesh, val=1, **kwargs):
    T = torch.zeros_like(mesh[...,-1]) + val
    return T


class WienerLoss(nn.Module):
    """The WienerLoss class implements the adaptive Wiener criterion, which
    aims to compare two data samples through a convolutional filter. The
    methodology is inspired by the paper `Adaptive Waveform Inversion:
    Theory`_ (Warner and Guasch, 2014) and is presented in 
    _Convolve and Conquer \: Data Comparison with Wiener Filters:
        https://arxiv.org/pdf/2311.06558 (Cruz et al, 2023)

    Args:
        method, optional
            "fft" for Fast Fourier Transform or "ldr" for the
            Levinson-Durbin recursion algorithm. Defaults to "fft".
            Levinson-Durbin recursion is not yet implemented.
        filter_dim, optional
            the dimensionality of the filter. This parameter should be
            upper-bounded by the dimensionality of the data. If data is
            3-dimensional and filter_dim is set to 2, one filter is computed
            per channel dimension assuming format [B, NC, H , W]. Current
            implementation only supports filter dimensions for 1D, 2D and 3D.
            Defaults to 2
        filter_scale, optional
            the scale of the filters compared to the size of the data.
            Defaults to 2
        reduction, optional
            specifies the reduction to apply to the output, "mean" or "sum".
            Defaults to mean
        mode, optional
            "forward" or "reverse" computation of the filter. For details of
            the difference, refer to the original paper. Default "reverse"
        penalty_function, optional
            the penalty function to apply to the filter. If None, the penalty
            function is the identity. Takes "identity", "gaussian" "trainable"
            or custom penalty function. If "trainable" is passed, a nn.Parameter
            is initialised filled with a constant value. In order to train, the
            class "self.parameters()" should be passed to a compatible optimiser.
            Default None
        std, optional
            the standard deviation of the gaussian when penalty_function="gaussian".
            Mean is always zero. Default 1e-4
        store_filters, optional
            whether to store the filters in memory, useful for debugging.
            Default False.
        lmbda, optional
            the stabilisation value to compute the filter in mode="fft". Understood 
            as a percentage of th RMS of the cross-correlation. " Default 1e-4.
        corr_norm, optional,
            whether to normalise the frequency spectrum by the cross-correlation RMS
            when using mode="fft". Default True.
        clamp_min, optional
            filters are clipped to this minimum value after computation. If
            None, operation is disabled. Default none
        input_shape, optional 
            shape of input for pre-computation of (C, H, W) no batch of 
            important variables. Required if penalty_function is 'trainable'
            
    .. _Convolve and Conquer \: Data Comparison with Wiener Filters:
        https://arxiv.org/pdf/2311.06558
        """
    def __init__(self, method="fft",
                 filter_dim=2,
                 filter_scale=2,
                 reduction="mean",
                 mode="reverse",
                 penalty_function=None,
                 store_filters=False,
                 lmbda=1e-4,
                 std=1e-4,
                 clamp_min=None,
                 corr_norm=True,
                 input_shape=None):

        super(WienerLoss, self).__init__()

        # Store arguments
        self.lmbda = lmbda
        self.std = std
        self.filter_scale = filter_scale
        self.penalty_function = penalty_function
        self.mode = mode
        self.clamp_min = clamp_min
        self.corr_norm = corr_norm

        # Check arguments
        if type(store_filters) == bool:
            self.store_filters = store_filters
        else:
            raise ValueError("store_filters receives a boolean value, "
                             "but found {}".format(store_filters))

        if reduction == "mean" or reduction == "sum":
            self.reduction = reduction
        else:
            raise ValueError("reduction must be 'mean' or "
                             "'sum' but found {}".format(reduction))

        if filter_dim in [1, 2, 3]:
            self.filter_dim = filter_dim
            self.dims = tuple([-i for i in range(self.filter_dim, 0, -1)])
        else:
            raise ValueError("Filter dimensions must be 1, 2 or 3"
                             ", but found {}".format(filter_dim))
            
        
        if method.lower() == "fft" or method.lower() == "ldr":
            self.method = method.lower()
            if self.method == "ldr":
                raise NotImplementedError("LDR method not yet implemented")
        else:
            raise ValueError("method must be 'fft' or 'ldr'"
                             ", but found {}".format(method))
            
        # Check trainable function inputs
        if penalty_function=="trainable": 
            if input_shape is None:
                raise ValueError("trainable penalty requires 'input_shape' to be passed")
            else:
                self.train_penalty = True
        else:
            self.train_penalty = False

        # Variables to store metadata
        self.delta = None
        self.filters = None
        self.W = None
        self.filter_shape=None
        
        # Filter shape and delta
        if input_shape is not None:
            self.filter_shape = self._get_filter_shape([1]+list(input_shape))
            self.delta = self._make_delta(shape=self.filter_shape[-self.filter_dim:])
        
        # Initialise penalty as trainable parameters if prompted
        if self.train_penalty:
            self.W = nn.Parameter(torch.ones(self.filter_shape) + 100., requires_grad=True)
            with torch.no_grad():  # Ensure this operation does not track gradients
                self.W.copy_(self.W / torch.sum(self.W))

    def _get_filter_shape(self, input_shape):
        if self.filter_dim == 1:
            _, n = input_shape
            fs = [self.filter_scale*n]
        elif self.filter_dim == 2:
            _, nc, h, w = input_shape
            fs = [nc, self.filter_scale*h, self.filter_scale*w]
        elif self.filter_dim == 3:
            _, nc, h, w = input_shape
            fs = [self.filter_scale*nc,
                  self.filter_scale*h,
                  self.filter_scale*w]

        # Make filter dimensions odd integers to allow spike at zero lag
        for i in range(len(fs)):
            fs[i] = int(fs[i])
            if fs[i] % 2 == 0:
                # Except nchannels for 2D filters, dimension
                # must match to input
                if (self.filter_dim == 2 and i == 0):
                    pass
                else:
                    fs[i] = fs[i] - 1
        return fs


    def _make_penalty(self, shape, eta=0., penalty_function=None, std=None, device="cpu"):
        arr = [torch.linspace(-1., 1., n, requires_grad=True)
               for n in shape]
        mesh = torch.meshgrid(arr, indexing="ij")
        mesh = torch.stack(mesh, axis=-1)
        if penalty_function is None or penalty_function == "identity":
            penalty = identity(mesh)
            
        elif penalty_function == "gaussian":
            std = self.std if std is None else std
            mean = torch.tensor([0. for i in range(mesh.shape[-1])], requires_grad=True)
            covmatrix = torch.diag(torch.tensor(
                [std**2 for i in range(mesh.shape[-1])], requires_grad=True))
            penalty = multigauss(mesh, mean, covmatrix)

        else:
            penalty = penalty_function(mesh)
        
        penalty = penalty + eta*torch.rand_like(penalty)
        return penalty.to(device)
    
    def _make_delta(self, shape):
        delta = torch.empty(shape)
        torch.nn.init.dirac_(delta.unsqueeze(0).unsqueeze(0))
        return delta


    def wienerfft(self, x, y, fs, lmbda=1e-9):
        """
        Calculates the optimal least squares convolutional Wiener filter that
        transforms signal x into signal y using FFT and a pre-whitening value
        lmbda.
        """
        assert x.shape == y.shape, "signals x and y must be the same shape"
        
        # Cross-correlation of x with y
        Fccorr = torch.fft.fftn(x, dim=self.dims)\
            * torch.conj(torch.fft.fftn(y, dim=self.dims))

        # Auto-correlation of x
        Facorr = torch.fft.fftn(x, dim=self.dims)\
            * torch.conj(torch.fft.fftn(x, dim=self.dims))
            
        # Normalise correlations
        if self.corr_norm:
            rms_ = rms(torch.abs(Fccorr))
            Fccorr = Fccorr / rms_
            Facorr = Facorr / rms_
       

        # Deconvolution of Fccorr by Facorr with relative pre-whitening
        lmbda = lmbda * rms(torch.abs(Fccorr)) 
        Fdconv = (Fccorr + lmbda) / (Facorr + lmbda)

        # Inverse Fourier transform
        rolled = torch.fft.irfftn(Fdconv, fs[-self.filter_dim:], dim=self.dims)

        # Unrolling
        rolling = tuple([int(-x.shape[i]/2) - 1
                        for i in range(1, len(x.shape), 1)])[-len(self.dims):]
        return torch.roll(rolled, rolling, dims=self.dims)

    def wienerldr(self, x, y, fs, lmbda=1e-9):
        """
        calculates the optimal least squares convolutional Wiener filter that
        transforms signal x into signal y using the ldr Toeplitz matrix
        implementation
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

                # Stabilize diagonals
                tmp = tmp + torch.diag(torch.zeros_like(torch.diagonal(tmp))
                                       + torch.abs(tmp).max()*lmbda)
                tmp = torch.inverse(tmp)
                v[i] = tmp @ (D_t @ pad_signal(y[i].unsqueeze(0),
                                                    [D_t.shape[1]])[0])

        elif self.filter_dim == 2:
            for i in range(bs):
                for j in range(x.shape[1]):
                    # Compute filter
                    Z = self.make_doubly_block(x[i][j])
                    Z_t = Z.T
                    tmp = Z_t @ Z

                    # Stabilize diagonals
                    tmp = tmp + torch.diag(torch.zeros_like(
                                           torch.diagonal(tmp)) +
                                           torch.abs(tmp).max()*self.lmbda)

                    tmp = torch.inverse(tmp)
                    tmp = tmp @ (Z_t @ pad_signal(y[i][j].unsqueeze(0),
                                                       (3*y.shape[2] - 2,
                                                       3*y.shape[3] - 2)
                                                       ).flatten(start_dim=0))
                    v[i][j] = tmp.reshape(fs[-self.filter_dim:])
        return v

    def forward(self, recon, target, lmbda=None, gamma=0., eta=0.):
        ''' The function takes in a reconstructed signal, a target signal,
        and a few other parameters, and returns the loss

        Args
            recon
                the reconstructed signal
            target
                the target signal
            lmbda, optional
                the stabilization value to compute the filter. If passed,
                overwrites the class attribute of same name. Default None.
            gamma, optional
                noise to add to both target and reconstructed signals
                for training stabilization. The noise added is different
                for target and reconstruction, and vary per class call.
                Default 0.
            eta, optional
                noise to add to penalty function. Default 0.

        '''

        assert recon.shape == target.shape, "recon and target must be of the" \
            "same shape but found {} and {}".format(recon.shape, target.shape)

        # White noise to recon and target for stabilization
        recon = recon + gamma * torch.rand_like(recon)
        target = target + gamma * torch.rand_like(target)

        # Batch size
        bs = recon.shape[0]

        # Flatten recon and target for 1D filters
        if self.filter_dim == 1:
            recon = recon.flatten(start_dim=1)
            target = target.flatten(start_dim=1)

        # Define size of the filter, reserve memory to store them if prompted
        if self.filter_shape is None:
            self.filter_shape = self._get_filter_shape(recon.shape)
        if self.store_filters:
            self.filters = torch.zeros([bs]+self.filter_shape).to(recon.device)

        # Compute wiener filter
        lmbda = self.lmbda if lmbda is None else lmbda
        if self.method == "fft":
            recon = pad_signal(recon, self.filter_shape)
            target = pad_signal(target, self.filter_shape)
            if self.mode == "reverse":
                v = self.wienerfft(target, recon, self.filter_shape, lmbda)
            elif self.mode == "forward":
                v = self.wienerfft(recon, target, self.filter_shape, lmbda)
        elif self.method == "ldr":
            if self.mode == "reverse":
                v = self.wienerldr(target, recon, self.filter_shape, lmbda)
            if self.mode == "forward":
                v = self.wienerldr(recon, target, self.filter_shape, lmbda)

        # Clamp filters
        if self.clamp_min is not None:
            v = torch.clamp(v, min=self.clamp_min)

        # Store filters if prompted
        if self.store_filters:
            self.filters = v[:]

        # Penalty function - recompute every iteration to recreate the computational graph
        if not self.train_penalty:
            self.W = self._make_penalty(
                shape=self.filter_shape[-self.filter_dim:],
                eta=eta, device=v.device,
                penalty_function=self.penalty_function
            )
        else:
            with torch.no_grad():
                pass
                self.W.clamp_(min=0.) # bound
                self.W.div_(torch.sum(self.W.data)) # normalise
                self.W.add_(torch.rand_like(self.W) * eta)  # stabilisie
        W = self.W.unsqueeze(0).expand_as(v).to(v.device)

        # Delta
        if self.delta is None:
            self.delta = self._make_delta(shape=self.filter_shape[-self.filter_dim:])
        delta = self.delta.unsqueeze(0).expand_as(v).to(v.device)
        
        # Evaluate Loss
        f = 0.5 * torch.norm(W * (v - delta), p=2, dim=self.dims)

        # Reduce
        if self.reduction == "sum":
            f = f.sum()
        elif self.reduction == "mean":
            f = f.mean()
        return f

