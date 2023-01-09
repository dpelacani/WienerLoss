import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from math import floor


class AWLoss(nn.Module):
    """The AWLoss class implements the adaptive Wiener criterion, which
    aims to compare two data samples through a convolutional filter. The
    methodology is inspired by the paper `Adaptive Waveform Inversion:
    Theory`_ (Warner and Guasch, 2014).

    A matching filter `w` can be computed such that it transforms
    a targetsignal `p` into the data `d` under an L2 norm principle:

    g = || p*w - d ||^2

    Let 'Z' be the Toeplitz matrix formulation of `p` such that the
    equation above is equivalent to:
    g = || Zw - d ||^2

    Minimizing this functional:
    dgdw = Z^T (Zw - d)
    dgdw --> 0 : w = (Z^T @ Z)^(-1) @ Z^T @ d

    To stabilize the matrix inversion, an amount is added to the diagonal
    of (Z^T @ Z) based on a value epsilon such that the inverted matrix is
    (Z^T @ Z) + max(diagonal(Z^T @ Z)) * epsilon


    In 2D, convolving p with w (or w with p) is equivalent to the matrix
    vector multiplication Zd where Z is the doubly block Toeplitz of the
    reconstructed image P and w is the flattened array of the 2D kernel W.

    Therefore, the system is equivalent to solving || Zw - d ||^2, and
    the solution to w is given by
    w = (Z^T @ Z + max(diagonal(Z^T @ Z)) * epsilon)^(-1) @ Z^T @ d

    This composes the direct method.

    Alternatively, convolution can be performed in the frequency domain
    with multiplication and division operations.
    This tends to be much more computationally efficient.

    The criterion is evaluated through a symmetrical monotonically decreasing
    function T and a dirac delta function that rewards when the filter kernel
    `w` is close to the identity kernel, and penalizes otherwise.

    f = 1/2 ||T * (v - delta)||^2

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
            the penalty function to apply to the filter. If None, a Gaussian
            penalty will be created of mean zero and standard deviations
            specified below. Mutually exclusive with "std". Default None
        store_filters, optional
            whether to store the filters in memory, useful for debugging.
            Option to store the filers before or after normalisation with
            "norm" and "unorm". Default False.
        epsilon, optional
            the stabilization value to compute the filter. Default 1e-4.
        std
            the standard deviation value of the zero-mean gaussian generated
            as a penalty function for the filter. If 'penalty_function' is
            passed this value will not be used. Default 1e-4.

    .. _Adaptive Waveform Inversion\: Theory:
        https://www.s-cube.com/media/1204/segam2014-03712e1.pdf

        """
    def __init__(self, method="fft", filter_dim=2, filter_scale=2,
                 reduction="mean", mode="reverse", penalty_function=None,
                 store_filters=False, epsilon=1e-4,  std=1e-4):

        super(AWLoss, self).__init__()

        # Store arguments
        self.epsilon = epsilon
        self.std = std
        self.filter_scale = filter_scale
        self.penalty_function = penalty_function
        self.mode = mode

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

        # Variables to store metadata
        self.filters = None
        self.T = None
        self.current_epoch = 0

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
        rv = -rv + rv.max()
        return rv

    def make_penalty(self, shape, std=1e-2, eta=0., penalty_function=None,
                     flip=False,  device="cpu"):
        arr = [torch.linspace(-1., 1., n, requires_grad=True).to(device)
               for n in shape]
        mesh = torch.meshgrid(arr)
        mesh = torch.stack(mesh, axis=-1)
        if penalty_function is None:
            mean = torch.tensor([0. for i in range(mesh.shape[-1])]).to(device)
            covmatrix = torch.diag(torch.tensor(
                [std**2 for i in range(mesh.shape[-1])])).to(device)
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
        # Cross-correlation of x with y
        Fccorr = torch.fft.fftn(torch.flip(x, self.dims), dim=self.dims)\
            * torch.fft.fftn(y, dim=self.dims)

        # Auto-correlation of x
        Facorr = torch.fft.fftn(torch.flip(x, self.dims), dim=self.dims)\
            * torch.fft.fftn(x, dim=self.dims)

        # Deconvolution of Fccorr by Facorr
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
        fs = self.get_filter_shape(recon.shape)
        if self.store_filters:
            self.filters = torch.zeros([bs]+fs).to(recon.device)

        # Compute wiener filter
        epsilon = self.epsilon if epsilon is None else epsilon
        if self.method == "fft":
            recon = self.pad_signal(recon, fs)
            target = self.pad_signal(target, fs)
            if self.mode == "reverse":
                v = self.wienerfft(target, recon, fs, epsilon)
            elif self.mode == "forward":
                v = self.wienerfft(recon, target, fs, epsilon)

        elif self.method == "direct":
            if self.mode == "reverse":
                v = self.wiener(target, recon, fs, epsilon)
            if self.mode == "forward":
                v = self.wiener(recon, target, fs, epsilon)

        # Normalise filter and store if prompted
        if self.store_filters == "unorm":
            self.filters = v[:]

        vnorm = torch.norm(v, p=2, dim=self.dims)
        for i in range(self.filter_dim):
            vnorm = vnorm.unsqueeze(-1)
        vnorm = vnorm.expand_as(v)

        if self.store_filters == "norm":
            self.filters = v[:] / vnorm

        # Penalty function
        self.T = self.make_penalty(shape=fs[-self.filter_dim:], std=self.std,
                                   eta=eta, device=recon.device, flip=True,
                                   penalty_function=self.penalty_function)
        T = self.T.unsqueeze(0).expand_as(v)

        # Delta
        self.delta = self.make_penalty(fs[-self.filter_dim:], std=3e-8,
                                       penalty_function=None,
                                       device=recon.device, flip=True)
        delta = self.delta.unsqueeze(0).expand_as(v)

        # Evaluate Loss
        f = 0.5 * torch.norm(T * (v - delta), p=2, dim=self.dims)

        f = f.sum()
        if self.reduction == "mean":
            f = f / recon.size(0)
        return f
