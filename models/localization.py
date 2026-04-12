
"""Localization modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import CustomDropout
from .vgg11 import VGG11

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        
        super(VGG11Localizer, self).__init__()

        self.encoder = VGG11(in_channels=in_channels)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.localization_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(512, 4)  # Final localization layer for bounding box coordinates
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box [B, 4] in normalized (cx, cy, w, h) format, values in [0, 1].
            Matches the ground-truth format produced by OxfordIIITPetDataset._mask_to_bbox.
        """
        x = self.encoder(x)                          # [B, 512, H/32, W/32]
        x = F.adaptive_avg_pool2d(x, (7, 7))         # [B, 512, 7, 7]
        out = self.localization_head(x)               # [B, 4]
        out = torch.sigmoid(out)    
        H, W = 224, 224  # Assuming input images are 224x224; adjust if different
        cx = out[:, 0] * W  # Scale cx by W
        cy = out[:, 1] * H  # Scale cy by H
        w = out[:, 2] * W  # Scale w by W
        h = out[:, 3] * H  # Scale h by H
        out = torch.stack([cx, cy, w, h], dim=1)  # [B, 4]
        # print("Pred range:", out.min().item(), out.max().item())
        return out                                    # (cx, cy, w, h) normalized


# x = torch.randn(2, 3, 224, 224)
# model = VGG11Localizer()
# out = model(x)
# print("Pred:", out[0])
# print(out)  
# print(out.shape)   # EXPECT: [2, 4]
# print(out.min(), out.max()) 
