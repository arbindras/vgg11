"""Dataset skeleton for Oxford-IIIT Pet.
"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    def __init__(self, root, split="train", transform=None, mask_transform=None):
        """
        Args:
            root (str): Root directory of dataset
            split (str): 'train', 'val', or 'test'
            transform: Image transforms
            mask_transform: Mask transforms
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform

        self.images_dir = os.path.join(root, "images")
        self.annotations_dir = os.path.join(root, "annotations")
        self.trimap_dir = os.path.join(root,"annotations", "trimaps")

        if split == "train":
            split_file = os.path.join(self.annotations_dir, "trainval.txt")
        elif split == "test":
            split_file = os.path.join(self.annotations_dir, "test.txt")
        elif split == "val":
            split_file = os.path.join(self.annotations_dir, "trainval.txt")  # Use trainval for val split
        else:
            raise ValueError("split must be train, val, or test")
        
        self.samples = []
        self.class_to_idx = {}

        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                image_id = parts[0]
                class_id = int(parts[1]) - 1

                img_path = os.path.join(self.images_dir, image_id + ".jpg")
                mask_path = os.path.join(self.trimap_dir, image_id + ".png")

                self.samples.append((image_id,img_path, mask_path, class_id))

            self.classes = sorted(list(set([label for _, _, _, label in self.samples])))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        if self.split in ["train", "val"]:
            # Split trainval into train and val
            train_samples, val_samples = train_test_split(
                self.samples,
                test_size=0.2,
                random_state=42,
                shuffle=True,
                stratify=[s[3] for s in self.samples]  # Stratify by class labels
                )
            self.samples = train_samples if self.split == "train" else val_samples

        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        image_id, img_path, mask_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        mask = torch.from_numpy(np.array(mask)).long() - 1  # Convert to 0-based and tensor
        if mask.dim() == 3:
            mask = mask.squeeze(0)  # Remove channel dim if present

        bbox = self._mask_to_bbox(mask)

        return image, label, bbox, mask
    
    def _mask_to_bbox(self, mask):
        """
        Compute bounding box from segmentation mask
        Output: (x_center, y_center, width, height)
        """
        pos = torch.nonzero(mask == 0)

        if pos.shape[0] == 0:
            return torch.zeros(4)

        y_min = pos[:, 0].min()
        y_max = pos[:, 0].max()
        x_min = pos[:, 1].min()
        x_max = pos[:, 1].max()
 
        x_center = (x_min + x_max).float() / 2
        y_center = (y_min + y_max).float() / 2
        width = (x_max - x_min + 1).float()
        height = (y_max - y_min + 1).float()

        H, W = mask.shape
        x_center /= W
        y_center /= H
        width /= W
        height /= H

        return torch.tensor([x_center, y_center, width, height], dtype=torch.float32)