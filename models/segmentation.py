"""Segmentation model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import CustomDropout
from .vgg11 import VGG11, ConvBlock

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super(VGG11UNet, self).__init__()

        ######### Encoder #########

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
        #self.pool5 = nn.MaxPool2d(2, 2)

        ######### Decoder #########

        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        ###### Segmentation head #########
        self.dropout = CustomDropout(dropout_p)
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        ###### Encoder #########
        x1 = self.enc1(x)
        p1 = self.pool1(x1)

        x2 = self.enc2(p1)
        p2 = self.pool2(x2)

        x3_1 = self.enc3_1(p2)
        x3_2 = self.enc3_2(x3_1)
        p3 = self.pool3(x3_2)

        x4_1 = self.enc4_1(p3)
        x4_2 = self.enc4_2(x4_1)
        p4 = self.pool4(x4_2)

        x5_1 = self.enc5_1(p4)
        x5_2 = self.enc5_2(x5_1)

        ###### Decoder #########
        d4 = self.up4(x5_2)
        d4 = torch.cat([d4, x4_2], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, x3_2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)

        out = self.dropout(d1)
        out = self.seg_head(out)

        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return out
    
# x = torch.randn(2, 3, 224, 224)
# model = VGG11UNet()
# out = model(x)

# print(out.shape)  # EXPECT: [2, 3, 224, 224]