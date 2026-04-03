import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset.stare_dataset import get_file_paths, create_folds, get_dataloaders
from models.unet_model import UNet
from utils.metrics import dice_score, dice_loss, iou_score, format_time
from utils.visualize import save_predictions
from utils.model_utils import load_model

# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "C:/1.lwBrown/Lee Lab/Evaluation_STARE/STARE/vessel_segmentation/images"
MASK_DIR  = "C:/1.lwBrown/Lee Lab/Evaluation_STARE/STARE/vessel_segmentation/masks_ah"

EPOCHS = 150
LR = 1e-5
weight_decay = 1e-5
BATCH_SIZE = 4
pos_weight = torch.tensor([3.0]).to(DEVICE)
##### basic learning hyperparameters####
EXPERIMENT_NAME = "unet_gn_leaky_lr1e-5_pos3_bce+dice_150epochs_wd1e-5_nodropout"
RESUME = True

BASE_SAVE_DIR = "C:/1.lwBrown/Lee Lab/Evaluation_STARE/experiments_unet"
BASE_MODEL_DIR = os.path.join(BASE_SAVE_DIR, "unet_base_train")

SAVE_DIR = os.path.join(BASE_SAVE_DIR, EXPERIMENT_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Saving results to: {SAVE_DIR}")

# -------------------------
# Loss
# -------------------------

bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# -------------------------
# Train One Epoch
# -------------------------
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# -------------------------
# Evaluate
# -------------------------
def evaluate(model, loader):
    model.eval()
    total_dice = 0
    total_iou = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            dice = dice_score(outputs, masks)
            iou  = iou_score(outputs, masks)

            total_dice += dice.item()
            total_iou  += iou.item()

    return total_dice / len(loader), total_iou / len(loader)


# -------------------------
# Main Training (Single Fold)
# -------------------------
def train_single_split():

    # Load data
    image_paths, mask_paths = get_file_paths(IMAGE_DIR, MASK_DIR)

    # Use only ONE fold
    folds = create_folds(image_paths, mask_paths, n_splits=5)
    train_images, train_masks, test_images, test_masks = folds[0]

    train_loader, test_loader = get_dataloaders(
        train_images, train_masks,
        test_images, test_masks,
        batch_size=BATCH_SIZE
    )

    # Model
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay= weight_decay)

    start_epoch = 0
    best_dice = 0

    if RESUME and os.path.exists(MODEL_PATH):
        print("Resuming training...")

        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

        print(f"Resumed from epoch {start_epoch}")



    best_dice = 0
    train_losses = []
    val_dices = []

    start_time = time.time()

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_dice, val_iou = evaluate(model, test_loader)

        train_losses.append(train_loss)
        val_dices.append(val_dice)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Loss: {train_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            save_path = os.path.join(SAVE_DIR, "model.pth")


            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_dice': best_dice,
            }, save_path)

        # Time estimation
        epoch_time = time.time() - epoch_start
        epochs_left = EPOCHS - (epoch + 1)
        remaining = epoch_time * epochs_left

        print(f"⏱ Epoch: {format_time(epoch_time)} | Remaining: {format_time(remaining)}")

    print(f"\nBest Dice: {best_dice:.4f}")

    # -------------------------
    # Plot curves
    # -------------------------
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_dices, label="Val Dice")
    plt.legend()
    plt.title("Training Curve (Fast Mode)")

    plt.savefig(os.path.join(SAVE_DIR, "curve.png"))
    plt.close()

    # -------------------------
    # Save predictions
    # -------------------------
    save_predictions(model, test_loader, DEVICE, SAVE_DIR, fold="fast+")

# -------------------------
# inference trained model
# -------------------------
def run_inference(model, loader, save_dir):
    from utils.visualize import save_predictions

    print("Running inference...")

    save_predictions(
        model=model,
        loader=loader,
        device=DEVICE,
        save_dir=save_dir,
        fold="inference",
        num_samples=5
    )

# -------------------------
# Run
# -------------------------
MODE = "train"   # change to "inference" or "train"
# MODEL_PATH = os.path.join(BASE_MODEL_DIR, "model.pth")
MODEL_PATH = os.path.join(BASE_MODEL_DIR, "model_100epoch.pth")

if __name__ == "__main__":

    if MODE == "train":
        train_single_split()

    elif MODE == "inference":
        # Load data
        image_paths, mask_paths = get_file_paths(IMAGE_DIR, MASK_DIR)
        folds = create_folds(image_paths, mask_paths, n_splits=5)

        # Use same split as training
        train_images, train_masks, test_images, test_masks = folds[0]

        _, test_loader = get_dataloaders(
            train_images, train_masks,
            test_images, test_masks,
            batch_size=1
        )

        # Load model
        model = load_model(MODEL_PATH)

        # Run inference
        run_inference(model, test_loader, SAVE_DIR)