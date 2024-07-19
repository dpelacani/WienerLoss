import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from math import floor


class WienerLoss(nn.Module):
    """The WienerLoss class implements the adaptive Wiener criterion, which
    aims to compare two data samples through a convolutional filter. The
    methodology is inspired by the paper `Adaptive Waveform Inversion:
    Theory`_ (Warner and Guasch, 2014) and is presented in 
    _Convolve and Conquer \: Data Comparison with Wiener Filters:
        https://arxiv.org/pdf/2311.06558 (Cruz et al, 2023)

    Args:
        method, optional
            "fft" for Fast Fourier Transform or "direct" for the
            Levinson-Durbin recurssion algorithm. Defaults to "fft"
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
            Mean is always zero. Default None
        store_filters, optional
            whether to store the filters in memory, useful for debugging.
            Option to store the filers before or after normalisation with
            "norm" and "unorm". Default False.
        epsilon, optional
            the stabilization value to compute the filter. Default 1e-4.
        rel_epsilon, optional
            whether the epislon value should be passed as absolute to the computation
            or understood by the class as a percentage of the root mean square
            of the cross-correlation. Defaults False.
        corr_norm, optional,
            if passed, it is the form of normalisation of the outputs of correlation
            in the filter computation. Takes "minmax", "rms", "ncc", and "zncc".
            Defaults to None.
        clamp_min, optional
            filters are clipped to this minimum value after computation. If
            None, operation is disabled. Default none
        input_shape, optional 
            shape of inpute for pre-computation of (C, H, W) no batch of 
            important variables. Required if penalty_function is 'trainable'
            
    .. _Convolve and Conquer \: Data Comparison with Wiener Filters:
        https://arxiv.org/pdf/2311.06558
        """
    def __init__(self, method="fft", filter_dim=2, filter_scale=2,
                 reduction="mean", mode="reverse", penalty_function=None,
                 store_filters=False, epsilon=1e-4,  std=1e-4, clamp_min=None,
                 corr_norm=None, rel_epsilon=False, input_shape=None):

        super(WienerLoss, self).__init__()

        # Store arguments
        self.epsilon = epsilon
        self.std = std
        self.filter_scale = filter_scale
        self.penalty_function = penalty_function
        self.mode = mode
        self.clamp_min = clamp_min
        self.rel_epsilon=rel_epsilon

        # Check arguments
        if store_filters in ["norm", "unorm"] or store_filters is False:
            self.store_filters = store_filters
        else:
            raise ValueError("store_filters must be 'norm', 'unorm' or"
                             "False, but found {}".format(store_filters))

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
            
        if corr_norm is not None and corr_norm.lower() in ["rms", "minmax", "ncc", "zncc"]:
            self.corr_norm = corr_norm.lower()
        elif corr_norm is None:
            self.corr_norm = corr_norm
        else:
            raise ValueError("Normalisation of cross correlation only"
                             " supports 'rms', 'minmax', 'ncc', and 'zncc' but found"
                             " '{}'".format(corr_norm))
        
        if method == "fft" or method == "direct":
            self.method = method
            if method == "direct":
                self.filter_scale = 2   # Larger filter scales not supported
                # for direct methods
                if self.filter_dim == 3:
                    raise NotImplementedError("3D filter implementation"
                                              "not available for the direct"
                                              "method")
        else:
            raise ValueError("method must be 'fft' or 'direct'"
                             ", but found {}".format(method))
            
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
        self.current_epoch = 0
        self.filter_shape=None
        
        # Filter shape and delta
        if input_shape is not None:
            self.filter_shape = self.get_filter_shape([1]+list(input_shape))
            self.delta = self.make_delta(shape=self.filter_shape[-self.filter_dim:])
        
        # Initialise penalty as trainable parameters if prompted
        if self.train_penalty:
            self.W = nn.Parameter(torch.ones(self.filter_shape) + 100., requires_grad=True)

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
        # each row will have a toeplitz
        # matrix of rowsize 3*X.shape[1] - 2
        r_block = 3 * X.shape[1] - 2

        # each row will have a toeplitz
        # matrix of colsize 2*X.shape[1] - 1
        c_block = 2*X.shape[1] - 1

        # how many rows / number of blocks
        n_blocks = X.shape[0]

        # total number of rows in doubly blocked toeplitz
        r = 3*(n_blocks * r_block) - 2 * r_block

        # total number of cols in doubly blocked toeplitz
        c = 2*(n_blocks * c_block) - 1 * c_block

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

    def pad_signal(self, x, shape, val=0):
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

    def multigauss(self, mesh, mean, covmatrix):
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
    
    def identity(self, mesh, val=1, **kwargs):
        if self.filter_dim == 1:
            xx = mesh[:,0]
        elif self.filter_dim == 2:
            xx, yy = mesh[:,:,0], mesh[:,:,1]
        elif self.filter_dim == 3:
            xx, yy, zz = mesh[:,:,:, 0], mesh[:,:,: 1], mesh[:,:,: 2]
        T = torch.zeros_like(xx) + val
        return T

    def make_penalty(self, shape, eta=0., penalty_function=None, std=None, device="cpu"):
        arr = [torch.linspace(-1., 1., n, requires_grad=True)
               for n in shape]
        mesh = torch.meshgrid(arr, indexing="ij")
        mesh = torch.stack(mesh, axis=-1)
        if penalty_function is None or penalty_function == "identity":
            penalty = self.identity(mesh)
            
        elif penalty_function == "gaussian":
            std = self.std if std is None else std
            mean = torch.tensor([0. for i in range(mesh.shape[-1])], requires_grad=True)
            covmatrix = torch.diag(torch.tensor(
                [std**2 for i in range(mesh.shape[-1])], requires_grad=True))
            penalty = self.multigauss(mesh, mean, covmatrix)

        else:
            penalty = penalty_function(mesh)
        
        penalty = penalty + eta*torch.rand_like(penalty)
        return penalty.to(device)
    
    def make_delta(self, shape):
        delta = torch.empty(shape)
        torch.nn.init.dirac_(delta.unsqueeze(0).unsqueeze(0))
        return delta


    def rms(self, x):
        square = torch.pow(x, 2)
        mean_square = torch.mean(square)#, dim=self.dims)
        rms = torch.sqrt(mean_square)
        return rms.item()

    def wienerfft(self, x, y, fs, prwh=1e-9):
        """
        George Strong (geowstrong@gmail.com)
        calculates the optimal least squares convolutional Wiener filter that
        transforms signal x into signal y using FFT
        """
        assert x.shape == y.shape, "signals x and y must be the same shape"
        
        # Cross-correlation of x with y
        Fccorr = torch.fft.fftn(x, dim=self.dims)\
            * torch.conj(torch.fft.fftn(y, dim=self.dims))

        # Auto-correlation of x
        Facorr = torch.fft.fftn(x, dim=self.dims)\
            * torch.conj(torch.fft.fftn(x, dim=self.dims))
            
        # Normalise correlations
        if self.corr_norm == "rms":
            rms = self.rms(torch.abs(Fccorr))
            Fccorr = Fccorr / rms
            Facorr = Facorr / rms
            
        elif self.corr_norm == "ncc":
            Fccorr = Fccorr / Fccorr.std()
            Facorr = Facorr / Facorr.std()
        
        elif self.corr_norm == "minmax":
            Fccorr = Fccorr / torch.max(torch.abs(Fccorr))
            Facorr = Facorr / torch.max(torch.abs(Facorr))
            
        elif self.corr_norm == "zncc":
            Fccorr = (Fccorr - Fccorr.mean()) / Fccorr.std()
            Facorr = (Facorr - Facorr.mean()) / Facorr.std()

        # Deconvolution of Fccorr by Facorr with (relative) pre-whitening
        if self.rel_epsilon:
            prwh = prwh * self.rms(torch.abs(Fccorr)) 
        Fdconv = (Fccorr + prwh) / (Facorr + prwh)

        # Inverse Fourier transform
        rolled = torch.fft.irfftn(Fdconv, fs[-self.filter_dim:], dim=self.dims)

        # Unrolling
        rolling = tuple([int(-x.shape[i]/2) - 1
                        for i in range(1, len(x.shape), 1)])[-len(self.dims):]
        return torch.roll(rolled, rolling, dims=self.dims)

    def wiener(self, x, y, fs, epsilon=1e-9):
        """
        calculates the optimal least squares convolutional Wiener filter that
        transforms signal x into signal y using the direct Toeplitz matrix
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
                                       + torch.abs(tmp).max()*epsilon)
                tmp = torch.inverse(tmp)
                v[i] = tmp @ (D_t @ self.pad_signal(y[i].unsqueeze(0),
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
                                           torch.abs(tmp).max()*self.epsilon)

                    tmp = torch.inverse(tmp)
                    tmp = tmp @ (Z_t @ self.pad_signal(y[i][j].unsqueeze(0),
                                                       (3*y.shape[2] - 2,
                                                       3*y.shape[3] - 2)
                                                       ).flatten(start_dim=0))
                    v[i][j] = tmp.reshape(fs[-self.filter_dim:])
        return v

    def forward(self, recon, target, epsilon=None, gamma=0., eta=0.):
        '''> The function takes in a reconstructed signal, a target signal,
        and a few other parameters, and returns the loss

        Args
            recon
                the reconstructed signal
            target
                the target signal
            epsilon, optional
                the stabilization value to compute the filter. If passed,
                overwrites the class attribute of same name. Default None.
            gamma, optional
                noise to add to both target and reconstructed signals
                for training stabilization. Default 0.
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
            self.filter_shape = self.get_filter_shape(recon.shape)
        if self.store_filters:
            self.filters = torch.zeros([bs]+self.filter_shape).to(recon.device)

        # Compute wiener filter
        epsilon = self.epsilon if epsilon is None else epsilon
        if self.method == "fft":
            recon = self.pad_signal(recon, self.filter_shape)
            target = self.pad_signal(target, self.filter_shape)
            if self.mode == "reverse":
                v = self.wienerfft(target, recon, self.filter_shape, epsilon)
            elif self.mode == "forward":
                v = self.wienerfft(recon, target, self.filter_shape, epsilon)

        elif self.method == "direct":
            if self.mode == "reverse":
                v = self.wiener(target, recon, self.filter_shape, epsilon)
            if self.mode == "forward":
                v = self.wiener(recon, target, self.filter_shape, epsilon)

        # Clamp filters
        if self.clamp_min is not None:
            v = torch.clamp(v, min=self.clamp_min)

        # Normalise filter and store if prompted
        if self.store_filters == "unorm":
            self.filters = v[:]

        vnorm = torch.norm(v, p=2, dim=self.dims)
        for i in range(self.filter_dim):
            vnorm = vnorm.unsqueeze(-1)
        vnorm = vnorm.expand_as(v)

        if self.store_filters == "norm":
            self.filters = v[:] / vnorm

        # Penalty function - recompute every iteration to recreate the computational graph
        if not self.train_penalty:
            self.W = self.make_penalty(
                shape=self.filter_shape[-self.filter_dim:],
                eta=eta, device=v.device,
                penalty_function=self.penalty_function
            )
        else:
            with torch.no_grad():
                self.W.add(torch.rand_like(self.W) * eta)   
                self.W.clamp_(min=0.5) #, max=1.)        
        W = self.W.unsqueeze(0).expand_as(v).to(v.device)


        # Delta
        if self.delta is None:
            self.delta = self.make_delta(shape=self.filter_shape[-self.filter_dim:])
        delta = self.delta.unsqueeze(0).expand_as(v).to(v.device)
        
        # Evaluate Loss
        f = 0.5 * torch.norm(W * (v - delta), p=2, dim=self.dims)

        # Reduce
        f = f.sum()
        if self.reduction == "mean":
            f = f / recon.size(0)
        return f
