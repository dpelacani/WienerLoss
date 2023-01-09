import torch
import numpy as np
from awloss import AWLoss


###############################################################################
#                             UTILITY FUNCTIONS                               #
###############################################################################

def square_img(shape, lag=(0, 0), fill_val=0, square_val=1., radius=1.):
    im = torch.zeros(shape) + fill_val

    xarr = torch.linspace(-shape[0], shape[0], shape[0])
    yarr = torch.linspace(-shape[1], shape[1], shape[1])

    lagx, lagy = lag

    idx = torch.where(xarr > 0 + lagx)[0][0]
    idy = torch.where(yarr > 0 + lagy)[0][0]
    im[idx-radius:idx+radius, idy-radius:idy+radius] = square_val

    return im


def make_delta(shape):
    dims = len(shape)
    delta = torch.zeros(shape)

    if dims == 1:
        assert torch.prod(torch.tensor(list(shape))) % 2 != 0
        idx_middle = int(((shape[0] - 1) / 2))
        delta[idx_middle] = 1.

    elif dims == 2:
        assert shape[0] % 2 != 0
        assert shape[1] % 2 != 0
        idx_x_middle = int(((shape[0] - 1) / 2))
        idx_y_middle = int(((shape[1] - 1) / 2))
        delta[idx_x_middle, idx_y_middle] = 1.

    elif dims == 3:
        assert shape[0] % 2 != 0
        assert shape[1] % 2 != 0
        assert shape[2] % 2 != 0
        idx_x_middle = int(((shape[0] - 1) / 2))
        idx_y_middle = int(((shape[1] - 1) / 2))
        idx_z_middle = int(((shape[2] - 1) / 2))
        delta[idx_x_middle, idx_y_middle, idx_z_middle] = 1.

    return delta


def unravel_index(index, shape):
    """https://discuss.pytorch.org/t/how-to-do-a-unravel-index-in-pytorch-
    just-like-in-numpy/12987/3"""
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def make_even(x):
    if x % 2 == 0:
        return x
    else:
        return x - 1


def make_odd(x):
    if x % 2 != 0:
        return x
    else:
        return x - 1


def peak(shape):
    return None


###############################################################################
#                          IDENTITY TEST FUNCTIONS                            #
###############################################################################

def test_identity(nc, awloss):
    # Create variable
    bs = torch.randint(size=(1,), low=1, high=8)
    n = torch.randint(size=(1,), low=2, high=32)
    x = torch.randn((bs, nc, n, n))

    # Evaluate loss, retrieve filters
    f = awloss(x, x)
    filters = awloss.filters
    filter_shape = filters.shape[1:]

    # Make delta functions for comparison
    if awloss.filter_dim in (1, 3):
        deltas = make_delta(filter_shape)

    elif awloss.filter_dim == 2:
        deltas = make_delta(filter_shape[1:])
        deltas = deltas.unsqueeze(0).repeat(nc, 1, 1)
    deltas = deltas.unsqueeze(0).expand_as(filters)

    # Check value of loss, and shape of filters
    assert torch.isclose(f, torch.tensor([0.]), atol=1e-5)
    assert torch.all(torch.isclose(filters, deltas, atol=1e-5))


def test_fft_1d_identity():
    nc = torch.randint(size=(1,), low=1, high=10)
    awloss = AWLoss(method="fft", filter_dim=1, filter_scale=4,
                    reduction="mean", mode="reverse", penalty_function=None,
                    store_filters="unorm", epsilon=0.,  std=1e-4)
    test_identity(nc, awloss)


def test_fft_2d_identity():
    nc = torch.randint(size=(1,), low=2, high=10)
    awloss = AWLoss(method="fft", filter_dim=2, filter_scale=4,
                    reduction="mean", mode="reverse", penalty_function=None,
                    store_filters="unorm", epsilon=0.,  std=1e-4)
    test_identity(nc, awloss)


def test_fft_3d_identity():
    nc = torch.randint(size=(1,), low=3, high=10)
    awloss = AWLoss(method="fft", filter_dim=3, filter_scale=4,
                    reduction="mean", mode="reverse", penalty_function=None,
                    store_filters="unorm", epsilon=0.,  std=1e-4)
    test_identity(nc, awloss)


def test_lr_1d_identity():
    nc = torch.randint(size=(1,), low=1, high=3)
    awloss = AWLoss(method="direct", filter_dim=1, filter_scale=4,
                    reduction="mean", mode="reverse", penalty_function=None,
                    store_filters="unorm", epsilon=0.,  std=1e-4)
    test_identity(nc, awloss)


def test_lr_2d_identity():
    nc = torch.randint(size=(1,), low=1, high=3)
    awloss = AWLoss(method="direct", filter_dim=2, filter_scale=4,
                    reduction="mean", mode="reverse", penalty_function=None,
                    store_filters="unorm", epsilon=0.,  std=1e-4)
    test_identity(nc, awloss)


###############################################################################
#                             LAG TEST FUNCTIONS                              #
###############################################################################

def test_lag(awloss):
    # Properties of the square image
    lag = list(torch.randint(size=(2,), low=-16, high=16))
    lag = torch.tensor([make_even(x) for x in lag])
    n = 64
    radius = 9

    # Create square images
    input = square_img((n, n), lag=lag, radius=radius)
    target = square_img((n, n), lag=(0, 0), radius=radius)

    # Batch and number of channels
    nc = 3
    input = input.unsqueeze(0).unsqueeze(0).repeat(1, nc, 1, 1)
    target = target.unsqueeze(0).unsqueeze(0).repeat(1, nc, 1, 1)

    # Evaluate loss, retrieve filters
    awloss(input, target)
    filters = awloss.filters
    filter_shape = filters.shape

    if awloss.filter_dim == 1:
        xarr = torch.linspace(-len(filters[0]), len(filters[0]),
                              len(filters[0]))
        peak = int(xarr[torch.argmax(torch.abs(filters[0])).item()])
        expected_peak = n*lag[0] + lag[1]
        assert peak == expected_peak.item()

    elif awloss.filter_dim == 2:
        xarr = np.linspace(-filter_shape[-2], filter_shape[-2],
                           filter_shape[-2])
        yarr = np.linspace(-filter_shape[-1], filter_shape[-1],
                           filter_shape[-1])

        peaky, peakx = torch.where(torch.abs(filters[0][0]) ==
                                   torch.max(torch.abs(filters[0][0])))
        peaky, peakx = int(yarr[peaky.item()]), int(xarr[peakx.item()])
        peak = torch.tensor([peaky, peakx])

        assert torch.all(peak == lag)

    elif awloss.filter_dim == 3:
        xarr = np.linspace(-filter_shape[-2], filter_shape[-2],
                           filter_shape[-2])
        yarr = np.linspace(-filter_shape[-1], filter_shape[-1],
                           filter_shape[-1])

        _, peaky, peakx = torch.where(torch.abs(filters[0]) ==
                                      torch.max(torch.abs(filters[0])))
        peaky = int(yarr[peaky.item()])
        peakx = int(xarr[peakx.item()])
        peak = torch.tensor([peaky, peakx])
        assert torch.all(peak == lag)


def test_fft_1d_lag():
    awloss = AWLoss(epsilon=3e-5, store_filters="unorm", method="fft",
                    filter_dim=1)
    test_lag(awloss)


def test_fft_2d_lag():
    awloss = AWLoss(epsilon=3e-5, store_filters="unorm", method="fft",
                    filter_dim=2)
    test_lag(awloss)


def test_fft_3d_lag():
    awloss = AWLoss(epsilon=3e-5, store_filters="unorm", method="fft",
                    filter_dim=3)
    test_lag(awloss)


if __name__ == "__main__":
    print("test_fft_1d_identity()")
    test_fft_1d_identity()
    print("done!")

    print("test_fft_2d_identity()")
    test_fft_2d_identity()
    print("done!")

    print("test_fft_3d_identity()")
    test_fft_3d_identity()
    print("done!")

    print("test_lr_1d_identity()")
    test_lr_1d_identity()
    print("done!")

    print("test_lr_2d_identity()")
    test_lr_2d_identity()
    print("done!")

    print("test_fft_1d_lag()")
    test_fft_1d_lag()
    print("done!")

    print("test_fft_2d_lag()")
    test_fft_2d_lag()
    print("done!")

    print("test_fft_3d_lag()")
    test_fft_3d_lag()
    print("done!")
