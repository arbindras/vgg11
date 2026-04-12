"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.reduction = reduction
        

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format.
        """
        def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
            """Convert (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)."""
            x_center, y_center, width, height = boxes.unbind(dim=-1)
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            return torch.stack([x_min, y_min, x_max, y_max], dim=-1)
        
        pred_boxes_xyxy = cxcywh_to_xyxy(pred_boxes)
        target_boxes_xyxy = cxcywh_to_xyxy(target_boxes)

        ix1 = torch.max(pred_boxes_xyxy[:, 0], target_boxes_xyxy[:, 0])
        iy1 = torch.max(pred_boxes_xyxy[:, 1], target_boxes_xyxy[:, 1])
        ix2 = torch.min(pred_boxes_xyxy[:, 2], target_boxes_xyxy[:, 2])
        iy2 = torch.min(pred_boxes_xyxy[:, 3], target_boxes_xyxy[:, 3])

        intersection_area = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
        pred_area = (pred_boxes_xyxy[:, 2] - pred_boxes_xyxy[:, 0]) * (pred_boxes_xyxy[:, 3] - pred_boxes_xyxy[:, 1])
        target_area = (target_boxes_xyxy[:, 2] - target_boxes_xyxy[:, 0]) * (target_boxes_xyxy[:, 3] - target_boxes_xyxy[:, 1])
        union_area = pred_area + target_area - intersection_area + self.eps

        iou = intersection_area / union_area
        loss = 1 - iou

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss