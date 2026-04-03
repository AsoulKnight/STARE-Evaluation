import os
import glob
import random
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Config
# =========================================================
class CFG:
    data_root = "DRIVE"
    img_size = 256
    batch_size = 4
    epochs = 60
    lr = 1e-4
    weight_decay = 1e-4
    num_workers = 2
    seed = 42
    val_ratio = 0.2
    base_ch = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "best_simple_transunet_drive.pth"
    pred_dir = "predictions_test"
    threshold = 0.5


# =========================================================
# Utils
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sorted_files(folder, exts):
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def read_rgb_image(path):
    # DRIVE images are usually .tif
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def read_gray_image(path):
    # Handles gif/png etc.
    img = imageio.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    return img


def normalize_image(img):
    img = img.astype(np.float32) / 255.0
    return img


def binarize_mask(mask, thr=127):
    return (mask > thr).astype(np.float32)


# =========================================================
# Dataset
# =========================================================
class DriveTrainDataset(Dataset):
    """
    For training/validation:
      images: training/images/*.tif
      manual: training/1st_manual/*.gif
      fov:    training/mask/*.gif   (optional)
    """
    def __init__(self, image_paths, manual_paths, fov_paths=None, img_size=256, augment=False):
        self.image_paths = image_paths
        self.manual_paths = manual_paths
        self.fov_paths = fov_paths
        self.img_size = img_size
        self.augment = augment

        assert len(self.image_paths) == len(self.manual_paths), \
            f"Mismatch: {len(self.image_paths)} images vs {len(self.manual_paths)} labels"

        if self.fov_paths is not None:
            assert len(self.image_paths) == len(self.fov_paths), \
                f"Mismatch: {len(self.image_paths)} images vs {len(self.fov_paths)} FOV masks"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = read_rgb_image(self.image_paths[idx])
        vessel = read_gray_image(self.manual_paths[idx])

        if self.fov_paths is not None:
            fov = read_gray_image(self.fov_paths[idx])
        else:
            fov = np.ones_like(vessel, dtype=np.uint8) * 255

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        vessel = cv2.resize(vessel, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        fov = cv2.resize(fov, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
                vessel = np.fliplr(vessel).copy()
                fov = np.fliplr(fov).copy()

            if np.random.rand() < 0.3:
                angle = np.random.uniform(-10, 10)
                center = (self.img_size // 2, self.img_size // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(
                    img, M, (self.img_size, self.img_size),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101
                )
                vessel = cv2.warpAffine(
                    vessel, M, (self.img_size, self.img_size),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
                fov = cv2.warpAffine(
                    fov, M, (self.img_size, self.img_size),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )

            if np.random.rand() < 0.3:
                alpha = np.random.uniform(0.9, 1.1)   # contrast
                beta = np.random.uniform(-0.05, 0.05) # brightness
                img = np.clip(img.astype(np.float32) * alpha / 255.0 + beta, 0, 1)
                img = (img * 255).astype(np.uint8)

        img = normalize_image(img)
        vessel = binarize_mask(vessel)
        fov = binarize_mask(fov)

        img = np.transpose(img, (2, 0, 1))       # HWC -> CHW
        vessel = np.expand_dims(vessel, axis=0)  # 1,H,W
        fov = np.expand_dims(fov, axis=0)        # 1,H,W

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(vessel, dtype=torch.float32),
            torch.tensor(fov, dtype=torch.float32),
        )


class DriveTestDataset(Dataset):
    """
    For inference only:
      images: test/images/*.tif
      fov:    test/mask/*.gif   (optional)
    """
    def __init__(self, image_paths, fov_paths=None, img_size=256):
        self.image_paths = image_paths
        self.fov_paths = fov_paths
        self.img_size = img_size

        if self.fov_paths is not None:
            assert len(self.image_paths) == len(self.fov_paths), \
                f"Mismatch: {len(self.image_paths)} images vs {len(self.fov_paths)} FOV masks"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = read_rgb_image(img_path)

        orig_h, orig_w = img.shape[:2]

        if self.fov_paths is not None:
            fov = read_gray_image(self.fov_paths[idx])
        else:
            fov = np.ones((orig_h, orig_w), dtype=np.uint8) * 255

        img_resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        fov_resized = cv2.resize(fov, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        img_resized = normalize_image(img_resized)
        fov_resized = binarize_mask(fov_resized)

        img_resized = np.transpose(img_resized, (2, 0, 1))
        fov_resized = np.expand_dims(fov_resized, axis=0)

        return (
            torch.tensor(img_resized, dtype=torch.float32),
            torch.tensor(fov_resized, dtype=torch.float32),
            img_path,
            orig_h,
            orig_w,
        )


# =========================================================
# Model
# =========================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = F.pad(
            x,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class TransformerBottleneck(nn.Module):
    def __init__(self, in_channels=256, embed_dim=256, num_heads=4, depth=2, ff_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_proj = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)   # [B, HW, C]
        tokens = self.transformer(tokens)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        x = self.out_proj(x)
        return x


class SimpleTransUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)

        self.trans = TransformerBottleneck(
            in_channels=base * 8,
            embed_dim=base * 8,
            num_heads=4,
            depth=2,
            ff_dim=base * 16,
        )

        self.up1 = Up(base * 8, base * 4, base * 4)
        self.up2 = Up(base * 4, base * 2, base * 2)
        self.up3 = Up(base * 2, base, base)

        self.outc = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x4 = self.trans(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


# =========================================================
# Loss + Metrics with FOV masking
# =========================================================
class MaskedBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, targets, valid_mask):
        # logits, targets, valid_mask: [B,1,H,W]
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        bce = (bce * valid_mask).sum() / (valid_mask.sum() + 1e-7)

        probs = torch.sigmoid(logits)
        probs = probs * valid_mask
        targets = targets * valid_mask

        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)

        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss


@torch.no_grad()
def masked_dice_score(logits, targets, valid_mask, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds * valid_mask
    targets = targets * valid_mask

    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()


@torch.no_grad()
def masked_iou_score(logits, targets, valid_mask, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds * valid_mask
    targets = targets * valid_mask

    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()


# =========================================================
# Train / Validate / Predict
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_meter = 0.0

    for imgs, targets, fov in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        fov = fov.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, targets, fov)
        loss.backward()
        optimizer.step()

        loss_meter += loss.item()

    return loss_meter / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    loss_meter = 0.0
    dice_meter = 0.0
    iou_meter = 0.0

    for imgs, targets, fov in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        fov = fov.to(device)

        logits = model(imgs)
        loss = criterion(logits, targets, fov)

        loss_meter += loss.item()
        dice_meter += masked_dice_score(logits, targets, fov, threshold=threshold)
        iou_meter += masked_iou_score(logits, targets, fov, threshold=threshold)

    n = max(len(loader), 1)
    return loss_meter / n, dice_meter / n, iou_meter / n


@torch.no_grad()
def predict_test(model, loader, device, out_dir, threshold=0.5):
    model.eval()
    ensure_dir(out_dir)

    for imgs, fov, img_paths, orig_h, orig_w in loader:
        imgs = imgs.to(device)
        fov = fov.to(device)

        logits = model(imgs)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        preds = preds * fov

        preds = preds.cpu().numpy()

        for i in range(preds.shape[0]):
            pred = preds[i, 0]
            h = int(orig_h[i])
            w = int(orig_w[i])

            pred = (pred * 255).astype(np.uint8)
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

            stem = Path(img_paths[i]).stem
            save_path = os.path.join(out_dir, f"{stem}_pred.png")
            imageio.imwrite(save_path, pred)


# =========================================================
# Main
# =========================================================
def main():
    seed_everything(CFG.seed)
    print("Device:", CFG.device)

    # -------- training files --------
    train_img_dir = os.path.join(CFG.data_root, "training", "images")
    train_manual_dir = os.path.join(CFG.data_root, "training", "1st_manual")
    train_fov_dir = os.path.join(CFG.data_root, "training", "mask")

    train_images = sorted_files(train_img_dir, ["*.tif", "*.TIF"])
    train_manuals = sorted_files(train_manual_dir, ["*.gif", "*.GIF"])
    train_fovs = sorted_files(train_fov_dir, ["*.gif", "*.GIF"]) if os.path.isdir(train_fov_dir) else None

    print(f"Found training images: {len(train_images)}")
    print(f"Found training manuals: {len(train_manuals)}")
    print(f"Found training FOV masks: {0 if train_fovs is None else len(train_fovs)}")

    assert len(train_images) > 0, "No training images found"
    assert len(train_images) == len(train_manuals), "Training images/manual labels count mismatch"
    if train_fovs is not None:
        assert len(train_images) == len(train_fovs), "Training images/FOV masks count mismatch"

    # -------- split train/val --------
    indices = list(range(len(train_images)))
    random.shuffle(indices)

    val_count = max(1, int(len(indices) * CFG.val_ratio))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    tr_images = [train_images[i] for i in train_idx]
    tr_manuals = [train_manuals[i] for i in train_idx]
    tr_fovs = [train_fovs[i] for i in train_idx] if train_fovs is not None else None

    va_images = [train_images[i] for i in val_idx]
    va_manuals = [train_manuals[i] for i in val_idx]
    va_fovs = [train_fovs[i] for i in val_idx] if train_fovs is not None else None

    print(f"Train split: {len(tr_images)}")
    print(f"Val split:   {len(va_images)}")

    train_ds = DriveTrainDataset(
        tr_images, tr_manuals, tr_fovs,
        img_size=CFG.img_size,
        augment=True
    )
    val_ds = DriveTrainDataset(
        va_images, va_manuals, va_fovs,
        img_size=CFG.img_size,
        augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True
    )

    # -------- model --------
    model = SimpleTransUNet(
        in_channels=3,
        out_channels=1,
        base=CFG.base_ch
    ).to(CFG.device)

    criterion = MaskedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay
    )

    best_dice = -1.0

    # -------- train --------
    for epoch in range(CFG.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CFG.device)
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, CFG.device, threshold=CFG.threshold
        )

        print(
            f"Epoch {epoch+1:03d}/{CFG.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f} | "
            f"val_iou={val_iou:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), CFG.save_path)
            print(f"  -> Saved best model to {CFG.save_path}")

    print(f"Best val Dice: {best_dice:.4f}")

    # -------- load best model --------
    model.load_state_dict(torch.load(CFG.save_path, map_location=CFG.device))

    # -------- test inference --------
    test_img_dir = os.path.join(CFG.data_root, "test", "images")
    test_fov_dir = os.path.join(CFG.data_root, "test", "mask")

    test_images = sorted_files(test_img_dir, ["*.tif", "*.TIF"])
    test_fovs = sorted_files(test_fov_dir, ["*.gif", "*.GIF"]) if os.path.isdir(test_fov_dir) else None

    print(f"Found test images: {len(test_images)}")
    print(f"Found test FOV masks: {0 if test_fovs is None else len(test_fovs)}")

    if len(test_images) > 0:
        test_ds = DriveTestDataset(
            test_images,
            test_fovs,
            img_size=CFG.img_size
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True
        )

        predict_test(
            model,
            test_loader,
            CFG.device,
            CFG.pred_dir,
            threshold=CFG.threshold
        )
        print(f"Saved test predictions to: {CFG.pred_dir}")


if __name__ == "__main__":
    main()