import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

from utils.preprocess import train_transform, val_transform


class STAREDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))

        # Load mask
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask = (mask > 0).astype(np.float32)

        # Apply transform (if any)
        if self.transform:
            image, mask = self.transform(image, mask)

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (C,H,W)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)       # (1,H,W)

        return image, mask



def get_file_paths(image_dir, mask_dir):
    image_files = sorted(os.listdir(image_dir))

    image_paths = []
    mask_paths = []

    for img_name in image_files:
        if img_name.endswith(".ppm"):
            image_path = os.path.join(image_dir, img_name)

            # build corresponding mask name
            mask_name = img_name.replace(".ppm", ".ah.ppm")
            mask_path = os.path.join(mask_dir, mask_name)

            # safety check
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")

            image_paths.append(image_path)
            mask_paths.append(mask_path)

    return image_paths, mask_paths

def create_folds(image_paths, mask_paths, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    folds = []

    for train_idx, test_idx in kf.split(image_paths):
        train_images = [image_paths[i] for i in train_idx]
        train_masks  = [mask_paths[i] for i in train_idx]

        test_images = [image_paths[i] for i in test_idx]
        test_masks  = [mask_paths[i] for i in test_idx]

        folds.append((train_images, train_masks, test_images, test_masks))

    return folds


def get_dataloaders(train_images, train_masks, test_images, test_masks, batch_size=4):
    # Dataset
    train_dataset = STAREDataset(
        train_images, train_masks,
        transform=train_transform
    )

    test_dataset = STAREDataset(
        test_images, test_masks,
        transform=val_transform
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader

image_dir = "C:/1.lwBrown/Lee Lab/Evaluation_STARE/STARE/vessel_segmentation/images"
mask_dir  = "C:/1.lwBrown/Lee Lab/Evaluation_STARE/STARE/vessel_segmentation/masks_ah"

# Get paths
image_paths, mask_paths = get_file_paths(image_dir, mask_dir)

# Create folds
folds = create_folds(image_paths, mask_paths)

# Example: use Fold 1
train_images, train_masks, test_images, test_masks = folds[0]

# Create loaders
train_loader, test_loader = get_dataloaders(
    train_images, train_masks,
    test_images, test_masks
)

# Check one batch
for images, masks in train_loader:
    print(images.shape)  # (B,3,H,W)
    print(masks.shape)   # (B,1,H,W)
    break