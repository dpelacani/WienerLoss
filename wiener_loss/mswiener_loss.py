import torch
import torch.nn as nn
import torch.nn.functional as F

from .wiener_loss import WienerLoss

class MSWienerLoss(nn.Module):
    def __init__(self, 
                 scales=5, 
                 weights=None, 
                 downsample='avg', 
                 filter_scale=2, 
                 reduction='mean', 
                 mode='reverse',
                 penalty_function=None,
                 store_filters=False,
                 lmbda=1e-4,
                 std=1e-4,
                 clamp_min=None,
                 corr_norm=True,
                 input_shape=None):
        """
        Multi-Scale Wiener Loss.

        Args:
            scales (int): Number of scales.
            weights (list or None): Weights for each scale.
            downsample (str): Method of downsampling ('avg' or 'bilinear').
            Other args: Passed to WienerLoss.
        """
        super(MSWienerLoss, self).__init__()

        self.scales = scales
        self.downsample = downsample
        if weights is None:
            self.weights = [1.0 / scales] * scales
        else:
            assert len(weights) == scales, "weights length must match number of scales"
            self.weights = weights
        
        # Get shapes for all scales
        if input_shape is not None:
            input_shapes = []
            h, w = input_shape[-2], input_shape[-1]
            for scale in range(self.scales):
                factor = 2 ** scale
                scaled_h = max(1, h // factor)
                scaled_w = max(1, w // factor)
                input_shapes.append((input_shape[0], scaled_h, scaled_w))

        self.wiener_losses = nn.ModuleList([
            WienerLoss(
                filter_scale=filter_scale,
                reduction=reduction,
                mode=mode,
                penalty_function=penalty_function,
                store_filters=store_filters,
                lmbda=lmbda,
                std=std,
                clamp_min=clamp_min,
                corr_norm=corr_norm,
                input_shape=input_shapes[scale] if input_shape is not None else None
            ) for _ in range(scales)
        ])

    def _downsample(self, x, scale):
        """Downsample input tensor by a factor of 2^scale."""
        factor = 2 ** scale
        if self.downsample == 'avg':
            return F.avg_pool2d(x, kernel_size=factor, stride=factor, ceil_mode=True)
        elif self.downsample == 'bilinear':
            size = [max(1, s // factor) for s in x.shape[-2:]]
            return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        else:
            raise ValueError("Unsupported downsample mode. Use 'avg' or 'bilinear'.")

    def forward(self, recon, target, **kwargs):
        """
        Compute the multi-scale Wiener loss.

        Args:
            recon (torch.Tensor): Reconstructed tensor of shape (B, C, H, W).
            target (torch.Tensor): Target tensor of shape (B, C, H, W).
            **kwargs: Additional arguments for WienerLoss forward method.

        Returns:
            torch.Tensor: Scalar loss.
        """
        total_loss = 0.0
        for scale in range(self.scales):
            # Downsample for current scale
            if scale > 0:
                recon_ds = self._downsample(recon, scale)
                target_ds = self._downsample(target, scale)
            else:
                recon_ds = recon
                target_ds = target

            # Apply WienerLoss at current scale
            loss = self.wiener_losses[scale](recon_ds, target_ds, **kwargs)
            weighted_loss = self.weights[scale] * loss
            total_loss += weighted_loss

        return total_loss