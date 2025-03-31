import torch
import pytest
import torch.nn.functional as F
from wiener_loss import WienerLoss, pad_signal   # replace 'your_module' with your actual module

@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("channels", [1, 2, 4])
@pytest.mark.parametrize("mode", ["forward", "reverse"])
@pytest.mark.parametrize("filter_scale", [1, 2])
def test_wiener_loss_basic_usage(dim, channels, filter_scale, mode):
    shape = [1, channels] + [33] * dim
    x = torch.rand(*shape, requires_grad=True)
    y = torch.rand_like(x, requires_grad=False)

    loss_fn = WienerLoss(filter_scale=filter_scale, corr_norm=True, mode=mode)
    loss = loss_fn(x, y)
    assert torch.isfinite(loss), "Loss is not finite"
    loss.backward()  # Check differentiability

@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("channels", [1, 2, 4])
@pytest.mark.parametrize("mode", ["forward", "reverse"])
@pytest.mark.parametrize("filter_scale", [1, 2])
def test_wiener_loss_identity(dim, channels, filter_scale, mode):
    shape = [1, channels] + [33] * dim
    x = torch.rand(*shape)

    loss_fn = WienerLoss(filter_scale=filter_scale, corr_norm=True, mode=mode)
    loss = loss_fn(x, x)
    assert loss.item() < 1e-5, f"Identity loss not close to zero: got {loss.item()}"

@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("channels", [1, 2, 4])
@pytest.mark.parametrize("mode", ["forward", "reverse"])
@pytest.mark.parametrize("filter_scale", [1, 2])
def test_wiener_loss_lag_detection(dim, channels, filter_scale, mode):
    size = 33
    lags = [0, 2, 5]
    for lag in lags:
        shape = [1, channels] + [size] * dim
        x = torch.zeros(*shape)

        # Delta in the centre for all channels
        center = [s // 2 for s in x.shape[2:]]
        for c in range(channels):
            x[(0, c) + tuple(center)] = 1.0

        # Shift in the last dimension
        x_lag = torch.roll(x, shifts=lag, dims=-1)

        loss_fn = WienerLoss(filter_scale=filter_scale, corr_norm=True, mode=mode)
        filter_shape = loss_fn._get_filter_shape(x.shape)

        x_padded = pad_signal(x, [1] + filter_shape)
        x_lag_padded = pad_signal(x_lag, [1] + filter_shape)

        wiener_filter = loss_fn.wienerfft(x_padded, x_lag_padded, filter_shape, lmbda=0.0)
        wiener_filter = torch.flip(wiener_filter, dims=(-1,))

        # Get effective lag
        for c in range(channels):
            w_c = wiener_filter[0, c]
            center_index = w_c.shape[-1] // 2
            lag_measured = torch.argmax(w_c.flatten()) % w_c.shape[-1] - center_index
            assert abs(lag_measured - lag) <= 1, f"Channel {c}: Expected lag {lag}, measured {lag_measured}"