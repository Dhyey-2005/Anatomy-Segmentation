import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SurgicalSegmentationDataset
from losses import DiceCELoss

# -------------------------
# BASIC CONFIG
# -------------------------

DATASET_ROOT = "DATASET"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 6
BATCH_SIZE = 4
EPOCHS = 60
LR = 1e-4
PATIENCE = 10

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------
# SIMPLE U-NET
# -------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.down1 = DoubleConv(3, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))

        bn = self.bottleneck(self.pool(d4))

        u4 = self.up4(bn)
        u4 = self.conv4(torch.cat([u4, d4], dim=1))

        u3 = self.up3(u4)
        u3 = self.conv3(torch.cat([u3, d3], dim=1))

        u2 = self.up2(u3)
        u2 = self.conv2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat([u1, d1], dim=1))

        return self.out(u1)

# -------------------------
# METRIC
# -------------------------

def mean_dice_score(logits, targets, num_classes):
    preds = torch.argmax(logits, dim=1)
    dice_scores = []

    for cls in range(1, num_classes):
        pred_cls = (preds == cls).float()
        tgt_cls = (targets == cls).float()

        inter = (pred_cls * tgt_cls).sum()
        union = pred_cls.sum() + tgt_cls.sum()

        if union > 0:
            dice_scores.append((2 * inter / (union + 1e-6)).item())

    return sum(dice_scores) / len(dice_scores) if dice_scores else 0.0

# -------------------------
# TRAINING LOOP
# -------------------------

def train():
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 70)
    print(f"TRAINING STARTED AT: {start_time}")
    print("=" * 70)

    print("\n[INFO] Loading datasets...")
    train_ds = SurgicalSegmentationDataset(DATASET_ROOT, "TRAIN")
    val_ds   = SurgicalSegmentationDataset(DATASET_ROOT, "VALIDATION")

    print(f"[INFO] Train samples      : {len(train_ds)}")
    print(f"[INFO] Validation samples : {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("\n[INFO] Initializing model...")
    model = UNet(NUM_CLASSES).to(DEVICE)
    print(f"[INFO] Model moved to device: {DEVICE}")

    criterion = DiceCELoss("class_weights.npy")
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("\n[INFO] Loss      : Dice + Weighted Cross Entropy")
    print("[INFO] Optimizer : Adam")
    print("[INFO] Metrics   : Mean Dice (no background)")
    print("\n" + "-" * 70)

    header = (
        "│ Epoch │ Train Loss │ Val Mean Dice │ Best Dice │ Patience │"
    )
    line = "├───────┼────────────┼──────────────┼───────────┼──────────┤"

    print("┌───────┬────────────┬──────────────┬───────────┬──────────┐")
    print(header)
    print(line)

    best_dice = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for batch_idx, (imgs, masks) in enumerate(train_loader, 1):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            try:
                outputs = model(imgs)
                loss, _ = criterion(outputs, masks)
            except Exception as e:
                print("\n❌ TRAINING ERROR DETECTED")
                print(f"Epoch {epoch} | Batch {batch_idx}")
                print(f"Mask unique values: {torch.unique(masks)}")
                raise e

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_dice = 0.0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                outputs = model(imgs)
                val_dice += mean_dice_score(outputs, masks, NUM_CLASSES)

        val_dice /= len(val_loader)

        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "unet_best.pth"))
            checkpoint_note = "✓"
        else:
            patience_counter += 1
            checkpoint_note = " "

        print(
            f"│ {epoch:^5} │ {train_loss:^10.4f} │ {val_dice:^14.4f} │ "
            f"{best_dice:^9.4f} │ {patience_counter}/{PATIENCE:^6} │"
        )

        if patience_counter >= PATIENCE:
            print("└───────┴────────────┴──────────────┴───────────┴──────────┘")
            print("\n[INFO] Early stopping triggered.")
            break

    print("└───────┴────────────┴──────────────┴───────────┴──────────┘")

    end_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 70)
    print(f"TRAINING ENDED AT: {end_time}")
    print(f"BEST VALIDATION DICE: {best_dice:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    train()