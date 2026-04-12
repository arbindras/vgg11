
"""Classification components
"""

import torch
import torch.nn as nn

from .layers import CustomDropout
from .vgg11 import VGG11
import torch.nn.functional as F

class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        super(VGG11Classifier, self).__init__()

        self.encoder = VGG11(in_channels=in_channels)

        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, num_classes)  # Final classification layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        x = self.encoder(x) # [B, 512, H/16, W/16]
        if x.shape[-1] != 7 or x.shape[-2] != 7:
            x = F.adaptive_avg_pool2d(x, (7, 7))  # Ensure the spatial dimensions are 7x7
        x = self.classifier_head(x) # [B, num_classes]
        return x
        

# x = torch.randn(2, 3, 224, 224)
# model = VGG11Classifier()
# out = model(x)

# print(out.shape)  # EXPECT: [2, 37]