import torch
import torch.nn.functional as F
import pytest
from math import floor, ceil
import numpy as np

from wiener_loss import WienerLoss  # Replace with actual import path if needed


def convolve_input_with_filter(input_tensor, wiener_filter):
    """Performs 2D convolution of input_tensor with wiener_filter batch-wise."""
    bs, c, h, w = input_tensor.shape
    pad_h = wiener_filter.shape[-2] // 2
    pad_w = wiener_filter.shape[-1] // 2
    padded_input = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    filtered_output = F.conv2d(padded_input, wiener_filter.unsqueeze(1), groups=bs)
    return filtered_output


@pytest.mark.parametrize("method", ["fft"])
@pytest.mark.parametrize("filter_dim", [2])
@pytest.mark.parametrize("reduction", ["mean", "sum"])


def test_wiener_loss_identity_input(method, filter_dim, reduction):
    torch.manual_seed(0)
    batch_size = 2
    channels = 1
    height = 32
    width = 32

    # Create identical recon and target tensors
    x = torch.randn(batch_size, channels, height, width, dtype=torch.float32)
    recon = x.clone().requires_grad_(True)
    target = x.clone().requires_grad_(False)

    loss_fn = WienerLoss(
        method=method,
        filter_dim=filter_dim,
        reduction=reduction,
        penalty_function="identity",
        input_shape=(channels, height, width)
    )

    loss = loss_fn(recon, target)
    assert isinstance(loss.item(), float), "Loss must return a scalar float value"
    assert loss.item() >= 0.0, "Loss must be non-negative"
    assert loss.item() < 1e-3, f"Loss too large for identical inputs: {loss.item()}"


def test_wiener_loss_random_input():
    torch.manual_seed(42)
    batch_size = 2
    channels = 1
    height = 16
    width = 16

    recon = torch.randn(batch_size, channels, height, width, dtype=torch.float32, requires_grad=True)
    target = torch.randn(batch_size, channels, height, width, dtype=torch.float32)

    loss_fn = WienerLoss(
        method="fft",
        filter_dim=2,
        reduction="mean",
        penalty_function="gaussian",
        std=1e-2,
        input_shape=(channels, height, width)
    )

    loss = loss_fn(recon, target)
    assert isinstance(loss.item(), float), "Loss must return a scalar float value"
    assert loss.item() >= 0.0, "Loss must be non-negative"
    loss.backward()
    assert recon.grad is not None, "Gradient not computed for recon"
    assert torch.all(torch.isfinite(recon.grad)), "Non-finite gradients detected"


def test_wiener_loss_gradient_check():
    torch.manual_seed(123)
    batch_size = 1
    channels = 1
    height = 8
    width = 8

    recon = torch.randn(batch_size, channels, height, width, dtype=torch.double, requires_grad=True)
    target = torch.randn(batch_size, channels, height, width, dtype=torch.double)

    loss_fn = WienerLoss(
        method="fft",
        filter_dim=2,
        reduction="mean",
        penalty_function="identity",
        input_shape=(channels, height, width)
    ).double()

    def func_to_check(x):
        return loss_fn(x, target)

    from torch.autograd import gradcheck
    gradcheck_result = gradcheck(func_to_check, (recon,), eps=1e-6, atol=1e-4)
    assert gradcheck_result, "Gradcheck failed for WienerLoss"


def test_wienerfft_filter_correctness():
    torch.manual_seed(123)
    batch_size = 2
    channels = 1
    height = 32
    width = 32

    x = torch.randn(batch_size, channels, height, width)
    y = torch.randn(batch_size, channels, height, width)

    loss_fn = WienerLoss(
        method="fft",
        filter_dim=2,
        reduction="mean",
        penalty_function="identity",
        input_shape=(channels, height, width)
    )

    # Pad signals to match filter shape
    filter_shape = loss_fn._get_filter_shape(x.shape)
    x_padded = F.pad(x, [s for pair in zip([floor((fs - d)/2) for fs, d in zip(filter_shape[-2:], x.shape[-2:])],
                                           [ceil((fs - d)/2) for fs, d in zip(filter_shape[-2:], x.shape[-2:])])
                      for s in pair], value=0)
    y_padded = F.pad(y, [s for pair in zip([floor((fs - d)/2) for fs, d in zip(filter_shape[-2:], y.shape[-2:])],
                                           [ceil((fs - d)/2) for fs, d in zip(filter_shape[-2:], y.shape[-2:])])
                      for s in pair], value=0)

    # Compute Wiener filter
    wiener_filter = loss_fn.wienerfft(x_padded, y_padded, filter_shape, lmbda=1e-4)

    # Apply filter to x
    filtered_output = convolve_input_with_filter(x_padded, wiener_filter)

    # Trim output to match original y shape
    trim_h = (filtered_output.shape[-2] - height) // 2
    trim_w = (filtered_output.shape[-1] - width) // 2
    filtered_output_trimmed = filtered_output[..., trim_h:trim_h+height, trim_w:trim_w+width]


    assert np.allclose(filtered_output_trimmed.detach().numpy(), y, atol=1e-5), "Wiener filter not correct"


if __name__ == "__main__":
    pytest.main([__file__])