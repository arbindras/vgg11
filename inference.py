
"""Inference and evaluation"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 📦 Model Loading
# =========================
def load_model(model_class: nn.Module, weight_path: str, **kwargs) -> nn.Module:
    """Load model weights from checkpoint."""
    model = model_class(**kwargs)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


# =========================
# 🔍 Predict functions
# =========================
@torch.no_grad()
def predict_classification(model: nn.Module, images: torch.Tensor):
    """
    Args:
        images: [B, C, H, W]
    Returns:
        preds: [B] numpy array
    """
    images = images.to(device)
    logits = model(images)              # [B, num_classes]
    preds = torch.argmax(logits, dim=1) # [B]
    return preds.cpu().numpy()


@torch.no_grad()
def predict_localization(model: nn.Module, images: torch.Tensor):
    """
    Args:
        images: [B, C, H, W]
    Returns:
        pred_boxes: [B, 4] in (x_center, y_center, width, height) format
    """
    images = images.to(device)
    pred_boxes = model(images)          # [B, 4]
    return pred_boxes.cpu().numpy()


@torch.no_grad()
def predict_segmentation(model: nn.Module, images: torch.Tensor):
    """
    Args:
        images: [B, C, H, W]
    Returns:
        pred_masks: [B, H, W] class index masks
    """
    images = images.to(device)
    logits = model(images)               # [B, num_classes, H, W]
    preds = torch.argmax(logits, dim=1)  # [B, H, W]
    return preds.cpu().numpy()


@torch.no_grad()
def predict_multitask(model: nn.Module, images: torch.Tensor):
    """
    Args:
        images: [B, C, H, W]
    Returns:
        dict with keys: "classification", "localization", "segmentation"
    """
    images = images.to(device)
    outputs = model(images)

    cls   = torch.argmax(outputs["classification"], dim=1)  # [B]
    boxes = outputs["localization"]                          # [B, 4]
    masks = torch.argmax(outputs["segmentation"], dim=1)    # [B, H, W]

    return {
        "classification": cls.cpu().numpy(),
        "localization":   boxes.cpu().numpy(),
        "segmentation":   masks.cpu().numpy(),
    }


# =========================
# 📐 Metric helpers
# =========================
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_classification_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute classification accuracy."""
    # FIX 4: numpy arrays have no .float() — use pure numpy
    return float((preds == labels).mean())


def compute_iou(pred_boxes: np.ndarray, target_boxes: np.ndarray, eps: float = 1e-6) -> float:
    """Compute mean IoU for bounding box predictions (cxcywh format)."""

    def cxcywh_to_xyxy(boxes):
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)

    pred   = cxcywh_to_xyxy(pred_boxes)
    target = cxcywh_to_xyxy(target_boxes)

    ix1 = np.maximum(pred[:, 0], target[:, 0])
    iy1 = np.maximum(pred[:, 1], target[:, 1])
    ix2 = np.minimum(pred[:, 2], target[:, 2])
    iy2 = np.minimum(pred[:, 3], target[:, 3])

    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    pred_area   = (pred[:, 2]   - pred[:, 0])   * (pred[:, 3]   - pred[:, 1])
    target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = pred_area + target_area - inter + eps

    return float((inter / union).mean())


def dice_score(pred_masks: np.ndarray, target_masks: np.ndarray, eps: float = 1e-6) -> float:
    """Compute mean Dice score for segmentation predictions."""
    pred_masks   = pred_masks.astype(bool)
    target_masks = target_masks.astype(bool)

    intersection = np.logical_and(pred_masks, target_masks).sum(axis=(1, 2))
    pred_sum     = pred_masks.sum(axis=(1, 2))
    target_sum   = target_masks.sum(axis=(1, 2))

    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    return float(dice.mean())


# =========================
# 📊 Evaluate functions
# =========================
def evaluate_classification(model: nn.Module, dataloader: torch.utils.data.DataLoader) -> float:
    """Evaluate classification accuracy."""
    model.eval()
    correct = 0
    total   = 0
    dev = next(model.parameters()).device

    with torch.no_grad():
        # FIX 1: unpack all 4 fields; boxes/masks ignored here
        for images, labels, boxes, masks in dataloader:
            images = images.to(dev)
            labels = labels.to(dev)
            logits = model(images)
            preds  = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return correct / total if total > 0 else 0.0


def evaluate_localization(model: nn.Module, dataloader: torch.utils.data.DataLoader) -> float:
    """Evaluate mean IoU for localization."""
    model.eval()
    ious = []
    dev = next(model.parameters()).device

    with torch.no_grad():
        # FIX 2: unpack all 4 fields; labels/masks ignored here
        for images, labels, target_boxes, masks in dataloader:
            images = images.to(dev)
            preds_np  = predict_localization(model, images)   # [B, 4]
            target_np = _to_numpy(target_boxes).copy()
            H, W = 224, 224
            target_np[:, 0] *= W
            target_np[:, 1] *= H
            target_np[:, 2] *= W
            target_np[:, 3] *= H
            ious.append(compute_iou(preds_np, target_np))
            # print("Pred sample:", preds_np[0])
            # print("GT sample:", target_np[0])

    return float(np.mean(ious)) if ious else 0.0


def evaluate_segmentation(model: nn.Module, dataloader: torch.utils.data.DataLoader) -> float:
    """Evaluate mean Dice score for segmentation."""
    model.eval()
    dice_scores = []
    dev = next(model.parameters()).device

    with torch.no_grad():
        # FIX 3: unpack all 4 fields; labels/boxes ignored here
        for images, labels, boxes, target_masks in dataloader:
            images    = images.to(dev)
            preds_np  = predict_segmentation(model, images)   # [B, H, W] numpy
            target_np = _to_numpy(target_masks)
            dice_scores.append(dice_score(preds_np, target_np))

    return float(np.mean(dice_scores)) if dice_scores else 0.0


def evaluate_multitask(model: nn.Module, dataloader: torch.utils.data.DataLoader) -> dict:
    """Evaluate all tasks for a multitask model."""
    model.eval()
    accs, ious, dices = [], [], []
    dev = next(model.parameters()).device

    with torch.no_grad():
        for images, labels, boxes, masks in dataloader:
            images    = images.to(dev)
            labels_np = _to_numpy(labels)
            boxes_np  = _to_numpy(boxes)
            masks_np  = _to_numpy(masks)

            outputs = predict_multitask(model, images)

            # Classification
            cls_out = outputs.get("classification")
            if cls_out is None:
                raise ValueError("predict_multitask must return 'classification'")
            accs.append(compute_classification_accuracy(_to_numpy(cls_out), labels_np))

            # Localization
            loc_out = outputs.get("localization")
            if loc_out is None:
                raise ValueError("predict_multitask must return 'localization'")
            ious.append(compute_iou(_to_numpy(loc_out), boxes_np))

            # Segmentation
            seg_out = outputs.get("segmentation")
            if seg_out is None:
                raise ValueError("predict_multitask must return 'segmentation'")
            dices.append(dice_score(_to_numpy(seg_out), masks_np))

    return {
        "classification_acc":  float(np.mean(accs))  if accs  else 0.0,
        "localization_iou":    float(np.mean(ious))   if ious  else 0.0,
        "segmentation_dice":   float(np.mean(dices))  if dices else 0.0,
    }