"""Robust training utilities for VGG11-based models"""

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from inference import (
    evaluate_classification,
    evaluate_localization,
    evaluate_multitask,
    evaluate_segmentation,
)
from losses.iou_loss import IoULoss


# =========================
# 🔧 Device
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 🌱 Seed
# =========================
def set_seed(
    seed: int = 42,
    cudnn_deterministic: bool = True,
    cudnn_benchmark: bool = False,
):
    """
    Make runs as reproducible as reasonably possible.

    Call this as early as possible, ideally before importing libraries that
    create CUDA contexts or spawn worker processes.

    Note: full bitwise determinism across different hardware, drivers, and
    PyTorch versions is not guaranteed. Enabling deterministic algorithms
    can raise errors for some ops and may slow training.
    """
    # 1) Python / hashing / numpy / random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 2) Torch CPU and CUDA seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 3) cuBLAS deterministic workspace
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # 4) cuDNN flags
    torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    # 5) Enforce PyTorch deterministic algorithms when available
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        try:
            torch.set_deterministic(True)
        except Exception:
            pass


def worker_init_fn(worker_id):
    """
    Use this in DataLoader to ensure each worker has a different but
    deterministic seed.
    Example: DataLoader(..., worker_init_fn=worker_init_fn)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =========================
# 💾 Checkpointing
# =========================
# def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric):
#     os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
#     torch.save(
#         {
#             "epoch": epoch,
#             "best_metric": best_metric,
#             "model_state": model.state_dict(),
#             "optimizer_state": optimizer.state_dict(),
#             "scheduler_state": scheduler.state_dict() if scheduler else None,
#         },
#         path,
#     )
def save_checkpoint(path, model, *args, **kwargs):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)

# =========================
# 📦 Loss Functions
# =========================
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        probs = torch.softmax(pred, dim=1)
        if target.dim() == 4:
            target = target.squeeze(dim=1)
        target = target.long()
        target_oh = nn.functional.one_hot(target, num_classes=pred.shape[1])
        target_oh = target_oh.permute(0, 3, 1, 2).float()

        intersection = (probs * target_oh).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + target_oh.sum(dim=(0, 2, 3))

        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()


# =========================
# 🔁 Generic Training Loop
# =========================
def _train_loop(
    model,
    train_loader,
    val_loader,
    loss_fn,
    metric_fn,
    epochs,
    optimizer,
    scheduler=None,
    clip_grad=1.0,
    checkpoint_path="model.pth",
    use_amp=True,
    maximize_metric=True,       # FIX 4: added so each task can control min/max
):
    model.to(DEVICE)

    # FIX 2: modern AMP API
    cuda_available = torch.cuda.is_available()
    amp_enabled = use_amp and cuda_available
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_metric = -float("inf") if maximize_metric else float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch in pbar:
            # Dataset yields (image, label, bbox, mask) — unpack all 4
            inputs, labels, boxes, masks = batch
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            boxes  = boxes.to(DEVICE, non_blocking=True)
            masks  = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            # FIX 2: modern autocast API
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(inputs)
                # FIX 1: always pass all 3 targets; each loss_fn decides which to use
                loss = loss_fn(outputs, labels, boxes, masks)

            scaler.scale(loss).backward()

            if clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=running_loss / num_batches)

        # Validation
        model.eval()
        with torch.no_grad():
            val_metric = metric_fn(model, val_loader)

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metric)
            else:
                scheduler.step()

        epoch_loss = running_loss / num_batches
        print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Val Metric={val_metric:.4f}")

        # FIX 4: respect maximize_metric flag
        is_better = (val_metric > best_metric) if maximize_metric else (val_metric < best_metric)
        if is_better:
            best_metric = val_metric
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, epoch, best_metric
            )
            print("✅ Saved best model")

    return best_metric


# =========================
# 🎯 Task-specific Trainers
# =========================

def train_classification(model, train_loader, val_loader, epochs=10, lr=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max")

    ce = nn.CrossEntropyLoss()

    # FIX 1 & 3: accept all 3 targets, use only labels, cast to long
    def loss_fn(out, labels, boxes, masks):
        return ce(out, labels.long())

    def metric_fn(m, loader):
        return evaluate_classification(m, loader)

    return _train_loop(
        model, train_loader, val_loader, loss_fn,
        metric_fn, epochs, optimizer, scheduler,
        checkpoint_path="classifier.pth",
        maximize_metric=True,
    )


def train_localization(model, train_loader, val_loader, epochs=10, lr=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max")

    iou = IoULoss()
    l1  = nn.SmoothL1Loss()

    mse = nn.MSELoss()
    def loss_fn(out, labels, boxes, masks):
        boxes = boxes.float()

        # 🔥 CONVERT NORMALIZED → PIXEL
        H, W = 224, 224

        boxes_pixel = boxes.clone()
        boxes_pixel[:, 0] *= W   # cx
        boxes_pixel[:, 1] *= H   # cy
        boxes_pixel[:, 2] *= W   # w
        boxes_pixel[:, 3] *= H   # h

        return 0.5*(mse(out, boxes_pixel) / (224 * 224)) + 0.5*iou(out, boxes_pixel)
    def metric_fn(m, loader):
        return evaluate_localization(m, loader)

    return _train_loop(
        model, train_loader, val_loader, loss_fn,
        metric_fn, epochs, optimizer, scheduler,
        checkpoint_path="localizer.pth",
        maximize_metric=True,
    )


def train_segmentation(model, train_loader, val_loader, epochs=10, lr=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max")

    ce   = nn.CrossEntropyLoss()
    dice = DiceLoss()

    # FIX 1: accept all 3 targets, use only masks; FIX 3: cast to long here
    def loss_fn(out, labels, boxes, masks):
        if masks.dim() == 4:
            masks = masks.squeeze(1)
        masks = masks.long()
        return ce(out, masks) + dice(out, masks)

    def metric_fn(m, loader):
        return evaluate_segmentation(m, loader)

    return _train_loop(
        model, train_loader, val_loader, loss_fn,
        metric_fn, epochs, optimizer, scheduler,
        checkpoint_path="segmenter.pth",
        maximize_metric=True,
    )


def train_multitask(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
    scheduler: Optional[Any] = None,
    save_best_path: Optional[str] = None,
    maximize_metric: bool = True,
    clip_grad: Optional[float] = 1.0,
):
    """
    Train a multitask model with classification, localization and segmentation outputs.

    Expects model to return a dict with keys:
        "classification", "localization", "segmentation".
    Expects DataLoader to yield batches of (inputs, labels, boxes, masks).
    """
    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    if scheduler is None:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max" if maximize_metric else "min"
        )

    # Loss modules
    ce   = nn.CrossEntropyLoss()
    l1   = nn.SmoothL1Loss()
    dice = DiceLoss()
    iou  = IoULoss()

    # FIX 2: modern AMP API
    cuda_available = torch.cuda.is_available()
    amp_enabled = use_amp and cuda_available
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    def loss_fn(
        out: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        boxes: torch.Tensor,
        masks: torch.Tensor,
    ):
        assert isinstance(out, dict), "Model must return dict for multitask outputs."
        required = {"classification", "localization", "segmentation"}
        missing = required - set(out.keys())
        if missing:
            raise KeyError(f"Model output missing required keys: {sorted(missing)}")

        # Targets already on device — just cast dtypes
        labels = labels.long()
        boxes  = boxes.float()
        masks  = masks.long()               # FIX 1: cast to long for CE loss

        loss_cls = ce(out["classification"], labels)
        loss_loc = iou(out["localization"], boxes) + l1(out["localization"], boxes)
        loss_seg = ce(out["segmentation"], masks) + dice(out["segmentation"], masks)

        total = loss_cls + loss_loc + loss_seg
        parts = {
            "loss_cls": loss_cls.detach(),
            "loss_loc": loss_loc.detach(),
            "loss_seg": loss_seg.detach(),
        }
        return total, parts

    best_metric = -float("inf") if maximize_metric else float("inf")
    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [], "val_metric": []
    }

    for epoch in range(1, epochs + 1):

        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            inputs, labels, boxes, masks = batch

            # FIX 3: move ALL targets to device in the loop
            inputs = inputs.to(device)
            labels = labels.to(device)
            boxes  = boxes.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=amp_enabled):   # FIX 2
                out = model(inputs)
                loss, _ = loss_fn(out, labels, boxes, masks)

            scaler.scale(loss).backward()

            if clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1

        avg_train_loss = running_loss / max(1, n_batches)
        history["train_loss"].append(avg_train_loss)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_running_loss = 0.0
        val_n   = 0
        correct = 0
        total   = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                inputs, labels, boxes, masks = batch

                # FIX 3: move ALL targets to device in the loop
                inputs = inputs.to(device)
                labels = labels.to(device)
                boxes  = boxes.to(device)
                masks  = masks.to(device)

                with torch.amp.autocast("cuda", enabled=amp_enabled):  # FIX 2
                    out = model(inputs)
                    loss, _ = loss_fn(out, labels, boxes, masks)

                val_running_loss += loss.item()
                val_n += 1

                preds = out["classification"].argmax(dim=1)
                correct += (preds == labels).sum().item()   # FIX 3: labels already on device
                total   += labels.numel()

        avg_val_loss = val_running_loss / max(1, val_n)
        val_accuracy = correct / total if total > 0 else 0.0

        history["val_loss"].append(avg_val_loss)
        history["val_metric"].append(val_accuracy)

        # ── Scheduler step ────────────────────────────────────────────────────
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_accuracy)
        elif hasattr(scheduler, "step"):
            try:
                scheduler.step()
            except TypeError:
                scheduler.step(epoch)

        # ── Save best checkpoint ──────────────────────────────────────────────
        is_better = (
            (val_accuracy > best_metric)
            if maximize_metric
            else (val_accuracy < best_metric)
        )
        if is_better:
            best_metric = val_accuracy
            if save_best_path:
                torch.save(model.state_dict(), save_best_path)
                print(f"  ✅ Saved best model → {save_best_path}")

        print(
            f"Epoch {epoch}/{epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  "
            f"val_acc={val_accuracy:.4f}  "
            f"best={best_metric:.4f}"
        )

    return model, history


# =========================
# 🚀 Main
# =========================
if __name__ == "__main__":
    # print("Training utilities ready. Plug in your dataloaders and models.")
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    from data.pets_dataset import OxfordIIITPetDataset
    from models.classification import VGG11Classifier
    from models.localization import VGG11Localizer
    from models.segmentation import VGG11UNet

    from train import (
        train_classification,
        train_localization,
        train_segmentation,
        set_seed,
    )

    # =========================
    # ⚙️ Config
    # =========================
    DATA_ROOT  = ""
    BATCH_SIZE = 64
    SEG_BATCH  = 16      # segmentation needs smaller batches (full feature maps in decoder)
    EPOCHS     = 15
    IMG_SIZE   = 224

    # =========================
    # 🌱 Seed
    # =========================
    set_seed(42)

    # =========================
    # 🖼️ Transforms
    # =========================
    image_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    # FIX 1: interpolation belongs inside Resize, not in Compose
    mask_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.NEAREST),
    ])

    # =========================
    # 📦 Datasets
    # =========================
    # FIX 2: OxfordIIITPetDataset already splits internally —
    #         create separate instances instead of random_split
    train_dataset = OxfordIIITPetDataset(
        root=DATA_ROOT,
        split="train",
        transform=image_transform,
        mask_transform=mask_transform,
    )
    val_dataset = OxfordIIITPetDataset(
        root=DATA_ROOT,
        split="val",
        transform=image_transform,
        mask_transform=mask_transform,
    )

    # =========================
    # 📊 DataLoaders
    # =========================
    # FIX 3: removed collate functions — default collate returns all 4 fields correctly
    # FIX 4: apply num_workers / pin_memory consistently to all loaders
    cuda_available = torch.cuda.is_available()
    loader_kwargs = dict(
            num_workers=4,
            pin_memory=cuda_available,
            persistent_workers=True,
        )

    # loader_kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True)

    train_loader_cls = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  **loader_kwargs)
    val_loader_cls   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    train_loader_loc = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  **loader_kwargs)
    val_loader_loc   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    train_loader_seg = DataLoader(train_dataset, batch_size=SEG_BATCH,  shuffle=True,  **loader_kwargs)
    val_loader_seg   = DataLoader(val_dataset,   batch_size=SEG_BATCH,  shuffle=False, **loader_kwargs)

    # =========================
    # 🔍 Quick sanity check
    # =========================
    batch = next(iter(train_loader_seg))
    images, labels, boxes, masks = batch
    print("images:", images.shape)        # expect [B, 3, 224, 224]
    print("masks: ", masks.shape)         # expect [B, 224, 224]
    print("mask values:", masks.unique()) # expect {-1, 0, 1} before loss (0-based after -1 shift in dataset)

    # =========================
    # 🎯 Task 1: Classification
    # =========================
    print("\n🚀 Training Classifier...\n")

    classifier = VGG11Classifier(num_classes=37)

    train_classification(
        classifier,
        train_loader_cls,
        val_loader_cls,
        epochs=EPOCHS,
        lr=1e-4,
    )

    # =========================
    # 📦 Task 2: Localization
    # =========================
    print("\n🚀 Training Localizer...\n")

    localizer = VGG11Localizer()

    train_localization(
        localizer,
        train_loader_loc,
        val_loader_loc,
        epochs=EPOCHS,
        lr=1e-4,
    )

    # =========================
    # 🎯 Task 3: Segmentation
    # =========================
    print("\n🚀 Training Segmenter...\n")

    segmenter = VGG11UNet(num_classes=3)

    train_segmentation(
        segmenter,
        train_loader_seg,
        val_loader_seg,
        epochs=EPOCHS,
        lr=1e-4,
    )

    print("\n✅ All models trained successfully!")
    