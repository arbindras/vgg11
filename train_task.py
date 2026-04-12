# import torch
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms
# from torchvision.transforms import InterpolationMode

# from data.pets_dataset import OxfordIIITPetDataset
# from models.classification import VGG11Classifier
# from models.localization import VGG11Localizer
# from models.segmentation import VGG11UNet

# from train import (
#     train_classification,
#     train_localization,
#     train_segmentation,
#     set_seed
# )

# # =========================
# # ⚙️ Config
# # =========================
# DATA_ROOT = ""
# BATCH_SIZE = 32
# EPOCHS = 15
# IMG_SIZE = 224

# # =========================
# # 🌱 Seed
# # =========================
# set_seed(42)

# # =========================
# # 🖼️ Transforms
# # =========================
# image_transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # For mask (no normalization)
# mask_transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.NEAREST),
# ])

# # =========================
# # 📦 Dataset
# # =========================
# dataset = OxfordIIITPetDataset(
#     root=DATA_ROOT,
#     split="train",
#     transform=image_transform,
#     mask_transform=mask_transform
# )


# # =========================
# # 📊 DataLoaders
# # =========================
# def collate_classification(batch):
#     images = torch.stack([b[0] for b in batch])
#     labels = torch.tensor([b[1] for b in batch])
#     return images, labels

# def collate_localization(batch):
#     images = torch.stack([b[0] for b in batch])
#     boxes = torch.stack([b[2] for b in batch])
#     return images, boxes

# def collate_segmentation(batch):
#     images = torch.stack([b[0] for b in batch])
#     masks = torch.stack([b[3] for b in batch])
#     return images, masks


# train_loader_cls = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader_cls = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# train_loader_loc = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader_loc = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# train_loader_seg = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
# val_loader_seg = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)




# batch = next(iter(train_loader_seg))
# images, labels, boxes, masks = batch
# print("images:", images.shape)   # expect [B, 3, 224, 224]
# print("masks: ", masks.shape)    # expect [B, 224, 224] ← if not, Cause 1 confirmed
# print("mask values:", masks.unique())  # expect {0, 1, 2} after -1 shift

# # =========================
# # 🎯 Task 1: Classification
# # =========================
# print("\n🚀 Training Classifier...\n")

# # classifier = VGG11Classifier(num_classes=37)

# # train_classification(
# #     classifier,
# #     train_loader_cls,
# #     val_loader_cls,
# #     epochs=EPOCHS,
# #     lr=1e-4
# # )


# # =========================
# # 📦 Task 2: Localization
# # =========================
# print("\n🚀 Training Localizer...\n")

# # localizer = VGG11Localizer()

# # train_localization(
# #     localizer,
# #     train_loader_loc,
# #     val_loader_loc,
# #     epochs=EPOCHS,
# #     lr=1e-4
# # )


# # =========================
# # 🎯 Task 3: Segmentation
# # =========================
# print("\n🚀 Training Segmenter...\n")

# segmenter = VGG11UNet(num_classes=3)

# train_segmentation(
#     segmenter,
#     train_loader_seg,
#     val_loader_seg,
#     epochs=EPOCHS,
#     lr=1e-4
# )

# print("\n✅ All models trained successfully!")

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
BATCH_SIZE = 32
SEG_BATCH  = 8      # segmentation needs smaller batches (full feature maps in decoder)
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
        num_workers=0,
        pin_memory=cuda_available,
        persistent_workers=False,
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

# localizer = VGG11Localizer()

# train_localization(
#     localizer,
#     train_loader_loc,
#     val_loader_loc,
#     epochs=EPOCHS,
#     lr=1e-4,
# )

# =========================
# 🎯 Task 3: Segmentation
# =========================
print("\n🚀 Training Segmenter...\n")

# segmenter = VGG11UNet(num_classes=3)

# train_segmentation(
#     segmenter,
#     train_loader_seg,
#     val_loader_seg,
#     epochs=EPOCHS,
#     lr=1e-4,
# )

print("\n✅ All models trained successfully!")