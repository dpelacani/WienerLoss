import torch
import torch.nn as nn
import torch.nn.functional as F

from .wiener_loss import WienerLoss

# TODO: make such that the code works without having to pass input_shape, i.e. it's inferred from recon and target
# TODO: add option to adaptive weights for each patch, based on patch energy or variance

class PatchWienerLoss(nn.Module):
    def __init__(self, 
                 input_shape,
                 patch_size=(16, 16), 
                 stride=(8, 8), 
                 filter_scale=2, 
                 reduction='mean',
                 mode='reverse',
                 penalty_function=None,
                 store_filters=False,
                 lmbda=1e-4,
                 std=1e-4,
                 clamp_min=None,
                 corr_norm=True,
                 global_weight=0.5):
        """
        Patched Wiener Loss with overlapping patches and hybrid global regularisation.

        Args:
            patch_size (tuple): Size of each patch (H, W).
            stride (tuple): Stride for overlapping patches.
            input_shape (tuple): Input shape (C, H, W).
            global_weight (float): Weighting factor for global Wiener loss.
            Other args: Passed to WienerLoss.
        """
        super(PatchWienerLoss, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.global_weight = min(1, max(0, global_weight))

        # Instantiate base WienerLoss for patches
        self.wiener_loss_patch = WienerLoss(
            filter_scale=filter_scale,
            reduction=reduction,
            mode=mode,
            penalty_function=penalty_function,
            store_filters=store_filters,
            lmbda=lmbda,
            std=std,
            clamp_min=clamp_min,
            corr_norm=corr_norm,
            input_shape=(input_shape[0], *patch_size)
        )
        
        # Instantiate base WienerLoss for global comparison
        self.wiener_loss_global = WienerLoss(
            filter_scale=filter_scale,
            reduction=reduction,
            mode=mode,
            penalty_function=penalty_function,
            store_filters=store_filters,
            lmbda=lmbda,
            std=std,
            clamp_min=clamp_min,
            corr_norm=corr_norm,
            input_shape=input_shape
        )

    def extract_patches(self, x):
        """
        Extract overlapping patches using unfold.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).
        
        Returns:
            Tensor: Patches of shape (B, num_patches, C, patch_H, patch_W).
        """
        B, C, H, W = x.shape
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)
        num_patches = patches.shape[-1]
        patches = patches.transpose(1, 2).reshape(B, num_patches, C, *self.patch_size)

        return patches

    def forward(self, recon, target, **kwargs):
        """
        Compute the Patched Wiener Loss.

        Args:
            recon (Tensor): Reconstructed image (B, C, H, W).
            target (Tensor): Ground truth image (B, C, H, W).
        
        Returns:
            Tensor: Combined loss.
        """
        B = recon.shape[0]
        
        # Extract overlapping patches
        recon_patches = self.extract_patches(recon)
        target_patches = self.extract_patches(target)

        # Reshape patches to (B * num_patches, C, patch_H, patch_W)
        recon_patches = recon_patches.reshape(-1, *recon_patches.shape[2:])
        target_patches = target_patches.reshape(-1, *target_patches.shape[2:])

        
        # Compute patch-wise Wiener Loss
        patch_loss = self.wiener_loss_patch(recon_patches, target_patches, **kwargs)

        # Compute global Wiener Loss
        global_loss = self.wiener_loss_global(recon, target, **kwargs)

        # Combine patch-wise and global loss
        total_loss = self.global_weight * global_loss + (1 - self.global_weight) * patch_loss

        return total_loss