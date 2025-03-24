import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from math import floor


def rms(x):
    square = torch.pow(x, 2)
    mean_square = torch.mean(square)
    rms = torch.sqrt(mean_square)
    return rms.item()


def pad_signal(x, shape, val=0):
    """
    Pad only spatial dimensions of shape [B, C, D1, D2, ...]
    """
    assert len(shape) == len(x.shape), f"Expected shape {len(shape)}, got {len(x.shape)}"

    pad = []
    for i in range(2, len(x.shape)):  # Skip batch and channel
        p1 = floor((shape[i] - x.shape[i]) / 2)
        p2 = shape[i] - x.shape[i] - p1
        pad.extend((p1, p2))

    pad = pad[::-1]  # Reverse for torch.nn.functional.pad
    return nn.functional.pad(x, pad, value=val)


def multigauss(mesh, mean, covmatrix):
    assert len(covmatrix.shape) == 2
    assert covmatrix.shape[0] == covmatrix.shape[1] == len(mean)
    assert len(mesh.shape) == len(mean) + 1

    rv = MultivariateNormal(mean, covmatrix)
    rv = torch.exp(rv.log_prob(mesh))
    rv = rv / torch.abs(rv).max()
    return rv


def identity(mesh, val=1, **kwargs):
    T = torch.zeros_like(mesh[..., -1]) + val
    return T


class WienerLoss(nn.Module):
    def __init__(self, method="fft",
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

        self.lmbda = lmbda
        self.std = std
        self.filter_scale = filter_scale
        self.penalty_function = penalty_function
        self.mode = mode
        self.clamp_min = clamp_min
        self.corr_norm = corr_norm

        if type(store_filters) == bool:
            self.store_filters = store_filters
        else:
            raise ValueError("store_filters must be boolean")

        if reduction in ["mean", "sum"]:
            self.reduction = reduction
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")

        if method.lower() not in ["fft", "ldr"]:
            raise ValueError("method must be 'fft' or 'ldr'")

        self.method = method.lower()
        if self.method == "ldr":
            raise NotImplementedError("LDR method not implemented")

        if penalty_function == "trainable":
            if input_shape is None:
                raise ValueError("input_shape required for trainable penalty")
            self.train_penalty = True
        else:
            self.train_penalty = False

        self.delta = None
        self.filters = None
        self.W = None
        self.filter_shape = None

        if input_shape is not None:
            self.filter_shape = self._get_filter_shape([1] + list(input_shape))
            self.delta = self._make_delta(shape=self.filter_shape[1:])

        if self.train_penalty:
            self.W = nn.Parameter(torch.ones(self.filter_shape) + 100., requires_grad=True)
            with torch.no_grad():
                self.W.copy_(self.W / torch.sum(self.W))

    def _get_filter_shape(self, input_shape):
        shape = input_shape[1:]  # Drop batch
        nchannels = shape[0]
        spatial_dims = shape[1:]

        fs_spatial = [self.filter_scale * dim for dim in spatial_dims]
        for i in range(len(fs_spatial)):
            fs_spatial[i] = int(fs_spatial[i])
            if fs_spatial[i] % 2 == 0:
                fs_spatial[i] -= 1

        return [nchannels] + fs_spatial

    def _make_penalty(self, shape, eta=0., penalty_function=None, std=None, device="cpu"):
        arr = [torch.linspace(-1., 1., n, requires_grad=True) for n in shape]
        mesh = torch.meshgrid(arr, indexing="ij")
        mesh = torch.stack(mesh, axis=-1)

        if penalty_function in [None, "identity"]:
            penalty = identity(mesh)
        elif penalty_function == "gaussian":
            std = self.std if std is None else std
            mean = torch.zeros(mesh.shape[-1], requires_grad=True)
            covmatrix = torch.diag(torch.tensor([std**2] * mesh.shape[-1], requires_grad=True))
            penalty = multigauss(mesh, mean, covmatrix)
        else:
            penalty = penalty_function(mesh)

        penalty = penalty + eta * torch.rand_like(penalty)
        return penalty.to(device)

    def _make_delta(self, shape):
        delta = torch.empty(shape)
        torch.nn.init.dirac_(delta.unsqueeze(0).unsqueeze(0))
        return delta

    def wienerfft(self, x, y, fs, lmbda=1e-9):
        assert x.shape == y.shape, "x and y must have the same shape"
        B, C = x.shape[:2]

        filter_dim = len(x.shape) - 2
        dims = tuple(range(2, 2 + filter_dim))

        Fx = torch.fft.fftn(x, dim=dims)
        Fy = torch.fft.fftn(y, dim=dims)
        Fccorr = Fx * torch.conj(Fy)
        Facorr = Fx * torch.conj(Fx)

        rms_ = rms(torch.abs(Fccorr))
        if self.corr_norm:
            Fccorr = Fccorr / rms_
            Facorr = Facorr / rms_

        lmbda_scaled = lmbda * rms_
        Fdconv = (Fccorr + lmbda_scaled) / (Facorr + lmbda_scaled)

        rolled = torch.fft.irfftn(Fdconv, fs[1:], dim=dims)
        rolling = tuple(-fs[i+1] // 2 for i in range(filter_dim))
        return torch.roll(rolled, rolling, dims=dims)

    def forward(self, recon, target, lmbda=None, gamma=0., eta=0.):
        assert recon.shape == target.shape, "recon and target must have the same shape"

        recon = recon + gamma * torch.rand_like(recon)
        target = target + gamma * torch.rand_like(target)
        bs = recon.shape[0]

        # Get filter dimension
        filter_dim = len(recon.shape) - 2
        dims = tuple(range(2, 2 + filter_dim))

        # Get filter shape
        if self.filter_shape is None:
            self.filter_shape = self._get_filter_shape(recon.shape)

        # Store filters
        if self.store_filters:
            self.filters = torch.zeros([bs] + self.filter_shape).to(recon.device)

        # Update lambda
        lmbda = self.lmbda if lmbda is None else lmbda

        # Pad signals to match filter shape
        recon = pad_signal(recon, [bs] + self.filter_shape)
        target = pad_signal(target, [bs] + self.filter_shape)

        # Wiener filter
        if self.method == "fft":
            if self.mode == "reverse":
                v = self.wienerfft(target, recon, self.filter_shape, lmbda)
            else:
                v = self.wienerfft(recon, target, self.filter_shape, lmbda)

        # Clamp
        if self.clamp_min is not None:
            v = torch.clamp(v, min=self.clamp_min)

        # Store filters
        if self.store_filters:
            self.filters = v[:]

        # Penalty
        if not self.train_penalty:
            self.W = self._make_penalty(
                shape=self.filter_shape[1:], eta=eta, device=v.device,
                penalty_function=self.penalty_function
            ).unsqueeze(0).unsqueeze(0).expand(bs, self.filter_shape[0], *self.filter_shape[1:])
        else:
            with torch.no_grad():
                # TODO: implement softmax for trainable penalty
                pass

        # Delta
        if self.delta is None:
            self.delta = self._make_delta(shape=self.filter_shape[-filter_dim:])
        delta = self.delta.unsqueeze(0).expand_as(v).to(v.device)

        f = 0.5 * torch.norm(self.W * (v - delta), p=2, dim=dims)

        if self.reduction == "sum":
            f = f.sum()
        elif self.reduction == "mean":
            f = f.mean()
        return f