import torch
from awloss import AWLoss


def test_fft_1d_identity():
    bs, nc = 4, 3
    w, h = torch.randint(low=2, high=100), torch.randint(low=2, high=100)
    x = torch.random((bs, nc, w, h))
    awloss = AWLoss(method="fft", filter_dim=1, filter_scale=2,
                 reduction="mean", mode="reverse", penalty_function=None,
                 store_filters=False, epsilon=1e-4,  std=1e-4)
