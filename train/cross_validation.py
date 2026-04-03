import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset.stare_dataset import get_file_paths, create_folds, get_dataloaders
from models.unet_model import UNet
from utils.metrics import dice_score, dice_loss, iou_score, format_time
from utils.visualize import save_predictions
import time
start_time = time.time()
# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "C:/1.lwBrown/Lee Lab/Evaluation_STARE/STARE/vessel_segmentation/images"
MASK_DIR  = "C:/1.lwBrown/Lee Lab/Evaluation_STARE/STARE/vessel_segmentation/masks_ah"

##### hyperparameters to change ####
EPOCHS = 50
LR = 5e-5
BATCH_SIZE = 4
NUM_FOLDS = 5

SAVE_DIR = "C:/1.lwBrown/Lee Lab/Evaluation_STARE/experiments"
os.makedirs(SAVE_DIR, exist_ok=True)



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

        #### modification able here #####
        pos_weight = torch.tensor([5.0]).to(DEVICE)  # try 3–10
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


#### saving results

results_path = os.path.join(SAVE_DIR, "results.txt")

with open(results_path, "w") as f:
    f.write("UNet 5-Fold Cross Validation Results\n\n")


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

    avg_dice = total_dice / len(loader)
    avg_iou  = total_iou / len(loader)

    return avg_dice, avg_iou


# -------------------------
# Train One Fold
# -------------------------
def train_fold(fold, folds):
    train_losses = []
    val_dices = []
    print(f"\n===== Fold {fold} =====")

    train_images, train_masks, test_images, test_masks = folds[fold]

    train_loader, test_loader = get_dataloaders(
        train_images, train_masks,
        test_images, test_masks,
        batch_size=BATCH_SIZE
    )

    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_dice = 0

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_dice, val_iou = evaluate(model, test_loader)

        train_losses.append(train_loss)
        val_dices.append(val_dice)

        print(f"[Fold {fold}] Epoch {epoch+1}/{EPOCHS}")
        print(f"Loss: {train_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            best_iou = val_iou
            save_path = os.path.join(SAVE_DIR, f"unet_fold{fold}.pth")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, save_path)

    print(f"Best Dice for Fold {fold}: {best_dice:.4f}")

    # plot
    plt.figure()

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_dices, label="Val Dice")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"Training Curve Fold {fold}")
    plt.legend()

    save_path = os.path.join(SAVE_DIR, f"fold{fold}_curve.png")
    plt.savefig(save_path)
    plt.close()

    # visualization

    save_predictions(model, test_loader, DEVICE, SAVE_DIR, fold)


    # time count

    epoch_time = time.time() - epoch_start
    elapsed_time = time.time() - start_time

    epochs_left = EPOCHS - (epoch + 1)
    estimated_remaining = epoch_time * epochs_left



    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Loss: {train_loss:.4f} | Dice: {val_dice:.4f}")
    print(f"⏱ Epoch Time: {format_time(epoch_time)} | Remaining: {format_time(estimated_remaining)}")

    return best_dice, best_iou

cv_start_time = time.time()
# -------------------------
# Main Cross Validation
# -------------------------
def run_cross_validation():
    image_paths, mask_paths = get_file_paths(IMAGE_DIR, MASK_DIR)
    folds = create_folds(image_paths, mask_paths, n_splits=NUM_FOLDS)
    fold_dice_results = []
    fold_iou_results = []

    for fold in range(NUM_FOLDS):
        best_dice, best_iou = train_fold(fold, folds)
        fold_dice_results.append(best_dice)
        fold_iou_results.append(best_iou)
        # save
        with open(results_path, "a") as f:
            f.write(f"Fold {fold}: Dice={best_dice:.4f}, IoU={best_iou:.4f}\n")

        # Time count
        elapsed = time.time() - cv_start_time
        folds_left = NUM_FOLDS - (fold + 1)

        avg_fold_time = elapsed / (fold + 1)
        remaining_time = avg_fold_time * folds_left

        print(f"⏱ Estimated CV remaining: {format_time(remaining_time)}")

    # Final results
    mean_dice = np.mean(fold_dice_results)
    std_dice  = np.std(fold_dice_results)
    mean_iou = np.mean(fold_iou_results)
    std_iou = np.std(fold_iou_results)

    print("\n===== FINAL RESULTS =====")
    print(f"Fold Dice Scores: {fold_dice_results}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Std Dice: {std_dice:.4f}")
    print(f"Fold IoU Scores: {fold_iou_results}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Std IoU: {std_iou:.4f}")\

    with open(results_path, "a") as f:
        f.write("\n")
        f.write(f"Mean Dice: {mean_dice:.4f}\n")
        f.write(f"Std Dice: {std_dice:.4f}\n")
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
        f.write(f"Std IoU: {std_iou:.4f}\n")



# -------------------------
# Run
# -------------------------
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