
"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Basic convolutional block:
    Conv2d → BatchNorm → ReLU

    Optionally supports stacking multiple conv layers (useful for VGG-style blocks).
    """

    def __init__(self, in_ch, out_ch, num_convs=1):
        super().__init__()

        layers = []
        for i in range(num_convs):
            layers.append(
                nn.Conv2d(
                    in_channels=in_ch if i == 0 else out_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    padding=1,
                    bias=False  # better with BatchNorm
                )
            )
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class VGG11(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super(VGG11, self).__init__()

        #Block 1
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)

        #Block 2
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)

        #Block 3
        self.enc3_1 = ConvBlock(128, 256)
        self.enc3_2 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)

        #Block 4
        self.enc4_1 = ConvBlock(256, 512)
        self.enc4_2 = ConvBlock(512, 512)
        self.pool4 = nn.MaxPool2d(2, 2)

        #Block 5
        self.enc5_1 = ConvBlock(512, 512)
        self.enc5_2 = ConvBlock(512, 512)
        self.pool5 = nn.MaxPool2d(2, 2)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """

        features = {}

        #Block 1
        x1 = self.enc1(x)
        features["enc1"] = x1
        x = self.pool1(x1)
        features["enc1_pooled"] = x

        #Block 2
        x2 = self.enc2(x)
        features["enc2"] = x2
        x = self.pool2(x2)
        features["enc2_pooled"] = x

        #Block 3
        x3_1 = self.enc3_1(x)
        x3_2 = self.enc3_2(x3_1)
        features["enc3_2"] = x3_2
        x = self.pool3(x3_2)
        features["enc3_2_pooled"] = x

        #Block 4
        x4_1 = self.enc4_1(x)
        x4_2 = self.enc4_2(x4_1)
        features["enc4_2"] = x4_2
        x = self.pool4(x4_2)
        features["enc4_2_pooled"] = x

        #Block 5
        x5_1 = self.enc5_1(x)
        x5_2 = self.enc5_2(x5_1)
        x5_pooled = self.pool5(x5_2)

        if return_features:
            return x5_pooled, features
        else:    
             return x5_pooled
        
VGG11Encoder = VGG11  # Alias for backward compatibility