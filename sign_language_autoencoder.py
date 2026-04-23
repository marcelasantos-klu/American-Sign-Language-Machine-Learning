import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

# =============================================================
#  American Sign Language Recognition — Convolutional Autoencoder
#  Chapter 07: Autoencoders and Anomaly Detection
#
#  The autoencoder is trained to reconstruct clean ASL images.
#  Anomaly detection works by measuring reconstruction error:
#    - Normal images (known ASL letters) → low error
#    - Anomalous images (corrupted / out-of-distribution) → high error
# =============================================================

# ── Reproducibility ───────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Config ────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
VAL_SPLIT = 0.15
PATIENCE = 5
LATENT_DIM = 128        # size of the bottleneck (compressed representation)
ANOMALY_THRESHOLD_PERCENTILE = 95   # flag top 5% reconstruction errors as anomalies
DATA_DIR = "data"
OUT_DIR = "outputs_autoencoder"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

# J=9 and Z=25 are motion-based and excluded from this dataset
DISPLAY_LABEL_MAP = {i: chr(ord('A') + i) for i in range(26) if i not in (9, 25)}


# =============================================================
# 1. DATASET
#    Note: the autoencoder is unsupervised — it only uses images,
#    not labels. Labels are kept so we can analyse results per class.
# =============================================================
class SignMNISTDataset(Dataset):
    """
    Sign Language MNIST dataset from CSV.
    Returns:
      image : float tensor [1, 28, 28]  — used as both input and target
      label : integer class index        — used only for result analysis
    """
    def __init__(self, dataframe, label_to_idx):
        self.raw_labels = dataframe["label"].values
        self.labels = torch.tensor(
            [label_to_idx[x] for x in self.raw_labels], dtype=torch.long
        )
        pixels = dataframe.drop("label", axis=1).values.astype(np.float32) / 255.0
        self.images = torch.tensor(pixels).reshape(-1, 1, 28, 28)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# =============================================================
# 2. MODEL — Convolutional Autoencoder
#
#  Encoder: compresses 1×28×28 → latent vector of size LATENT_DIM
#  Decoder: reconstructs latent vector → 1×28×28
#
#  The bottleneck forces the model to learn a compact representation
#  of what a "normal" ASL hand sign looks like. Images that don't
#  fit this learned structure will have high reconstruction error.
# =============================================================
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        # Encoder: 1×28×28 → 64×3×3 → latent_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 1×28×28  → 32×28×28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # 32×28×28 → 32×14×14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32×14×14 → 64×14×14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # 64×14×14 → 64×7×7

            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 64×7×7   → 64×7×7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # 64×7×7   → 64×3×3
        )
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, latent_dim),
            nn.ReLU(inplace=True),
        )

        # Decoder: latent_dim → 64×3×3 → 1×28×28
        #
        # ConvTranspose2d output size formula:
        #   out = (in - 1) * stride - 2 * padding + kernel_size
        #
        # Step 1: 3  → (3-1)*2  + 3 = 7   (k=3, stride=2, padding=0)
        # Step 2: 7  → (7-1)*2  + 2 = 14  (k=2, stride=2, padding=0)
        # Step 3: 14 → (14-1)*2 + 2 = 28  (k=2, stride=2, padding=0)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 3 * 3),
            nn.ReLU(inplace=True),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=0),  # 3×3  → 7×7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),  # 7×7  → 14×14
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, padding=0),   # 14×14 → 28×28
            nn.Sigmoid(),  # output in [0, 1] to match normalised pixel values
        )

    def encode(self, x):
        return self.encoder_fc(self.encoder_conv(x))

    def decode(self, z):
        x = self.decoder_fc(z).reshape(-1, 64, 3, 3)
        return self.decoder_conv(x)

    def forward(self, x):
        return self.decode(self.encode(x))


# =============================================================
# 3. LOAD DATA
# =============================================================
print("\n" + "=" * 70)
print("1 · LOADING DATA")
print("=" * 70)

train_df = pd.read_csv(os.path.join(DATA_DIR, "sign_mnist_train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "sign_mnist_test.csv"))

raw_unique_labels = sorted(train_df["label"].unique())
label_to_idx = {label: idx for idx, label in enumerate(raw_unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
class_names = [DISPLAY_LABEL_MAP[lbl] for lbl in raw_unique_labels]
num_classes = len(raw_unique_labels)

print(f"Number of classes      : {num_classes}")
print(f"Image size             : 28x28 grayscale")
print(f"Train CSV samples      : {len(train_df):,}")
print(f"Test CSV samples       : {len(test_df):,}")
print(f"Latent dimension       : {LATENT_DIM}")

full_train = SignMNISTDataset(train_df, label_to_idx)
test_dataset = SignMNISTDataset(test_df, label_to_idx)

indices = torch.randperm(
    len(full_train), generator=torch.Generator().manual_seed(SEED)
).tolist()
n_val = int(len(indices) * VAL_SPLIT)
val_indices = indices[:n_val]
train_indices = indices[n_val:]

train_dataset = Subset(full_train, train_indices)
val_dataset = Subset(full_train, val_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train split samples    : {len(train_dataset):,}")
print(f"Validation samples     : {len(val_dataset):,}")
print(f"Test samples           : {len(test_dataset):,}")


# =============================================================
# 4. TRAINING
# =============================================================
print("\n" + "=" * 70)
print("2 · TRAINING AUTOENCODER")
print("=" * 70)

model = ConvAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

# MSE loss: measures pixel-level reconstruction quality
criterion = nn.MSELoss()

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters       : {n_params:,}")
print(f"Loss function          : MSELoss (pixel reconstruction)")
print(f"Optimizer              : Adam (lr={LR})")

history = {"train_loss": [], "val_loss": []}
best_val_loss = float("inf")
patience_ctr = 0
best_weights = None

import copy
best_weights = copy.deepcopy(model.state_dict())

for epoch in range(1, EPOCHS + 1):
    # ── Train ──
    model.train()
    train_loss, total = 0.0, 0
    for imgs, _ in train_loader:          # labels ignored during training
        imgs = imgs.to(DEVICE)
        optimizer.zero_grad()
        recon = model(imgs)
        loss = criterion(recon, imgs)     # target = input (reconstruction)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
    train_loss /= total

    # ── Validate ──
    model.eval()
    val_loss, total = 0.0, 0
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(DEVICE)
            recon = model(imgs)
            val_loss += criterion(recon, imgs).item() * imgs.size(0)
            total += imgs.size(0)
    val_loss /= total
    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train loss={train_loss:.6f} | "
        f"val loss={val_loss:.6f} | "
        f"lr={current_lr:.6f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = copy.deepcopy(model.state_dict())
        patience_ctr = 0
        print(f"  New best val loss: {best_val_loss:.6f}")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_weights)
torch.save(model.state_dict(), f"{OUT_DIR}/autoencoder_best.pth")
print(f"\nBest validation loss   : {best_val_loss:.6f}")


# =============================================================
# 5. ANOMALY DETECTION
#
#  Strategy: compute per-image reconstruction error (MSE) on the
#  test set. A high error means the model struggled to reconstruct
#  the image — it looks "unusual" compared to training data.
#
#  To demonstrate detection, we inject synthetic anomalies:
#  heavily corrupted versions of real test images that should
#  score much higher reconstruction error than clean images.
# =============================================================
print("\n" + "=" * 70)
print("3 · ANOMALY DETECTION")
print("=" * 70)

model.eval()

# ── Compute reconstruction errors on the clean test set ──────
all_errors, all_labels, all_originals, all_recons = [], [], [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        recon = model(imgs)
        # Per-image MSE: mean over all pixels
        errors = ((recon - imgs) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
        all_errors.extend(errors)
        all_labels.extend(labels.numpy())
        all_originals.append(imgs.cpu())
        all_recons.append(recon.cpu())

all_errors = np.array(all_errors)
all_labels = np.array(all_labels)
all_originals = torch.cat(all_originals)
all_recons = torch.cat(all_recons)

# ── Inject synthetic anomalies ────────────────────────────────
# Take a subset of test images and corrupt them heavily with noise
# so we can verify the autoencoder correctly flags them
np.random.seed(SEED)
n_anomalies = 200
anomaly_indices = np.random.choice(len(test_dataset), n_anomalies, replace=False)
anomaly_imgs = all_originals[anomaly_indices]

# Corruption: strong salt-and-pepper noise + random block masking
corrupted_imgs = all_originals[anomaly_indices].clone()
noise = torch.rand_like(corrupted_imgs)
corrupted_imgs[noise < 0.15] = 0.0   # pepper
corrupted_imgs[noise > 0.85] = 1.0   # salt
# Random block mask (covers roughly 25% of the image)
for i in range(len(corrupted_imgs)):
    r = np.random.randint(0, 14)
    c = np.random.randint(0, 14)
    corrupted_imgs[i, :, r:r+14, c:c+14] = torch.rand(1).item()

with torch.no_grad():
    corrupted_recon = model(corrupted_imgs.to(DEVICE)).cpu()
anomaly_errors = ((corrupted_recon - corrupted_imgs) ** 2).mean(dim=[1, 2, 3]).numpy()

# ── Threshold: 95th percentile of clean test errors ──────────
threshold = np.percentile(all_errors, ANOMALY_THRESHOLD_PERCENTILE)
flagged_clean = (all_errors > threshold).sum()
flagged_anomalies = (anomaly_errors > threshold).sum()

print(f"Reconstruction error (clean test set):")
print(f"  Mean    : {all_errors.mean():.6f}")
print(f"  Std     : {all_errors.std():.6f}")
print(f"  Min     : {all_errors.min():.6f}")
print(f"  Max     : {all_errors.max():.6f}")
print(f"\nAnomaly threshold ({ANOMALY_THRESHOLD_PERCENTILE}th percentile): {threshold:.6f}")
print(f"\nClean images flagged as anomalous  : {flagged_clean} / {len(all_errors)} ({100*flagged_clean/len(all_errors):.1f}%)")
print(f"Corrupted images flagged as anomalous: {flagged_anomalies} / {n_anomalies} ({100*flagged_anomalies/n_anomalies:.1f}%)")


# =============================================================
# 6. PLOTS
# =============================================================
print("\n" + "=" * 70)
print("4 · SAVING PLOTS")
print("=" * 70)

# ── Training loss curves ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0f0f1a")
ax.set_facecolor("#0f0f1a")
ep = range(1, len(history["train_loss"]) + 1)
ax.plot(ep, history["train_loss"], color="#e040fb", linewidth=2, label="Train Loss")
ax.plot(ep, history["val_loss"], color="#00e5ff", linewidth=2, linestyle="--", label="Val Loss")
ax.set_title("Autoencoder Training Loss (MSE)", color="white", fontsize=15, fontweight="bold")
ax.set_xlabel("Epoch", color="#aaaacc")
ax.set_ylabel("MSE Loss", color="#aaaacc")
ax.tick_params(colors="white")
ax.grid(color="#222244", linewidth=0.5)
for spine in ax.spines.values():
    spine.set_edgecolor("#333355")
ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/1_training_loss.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print(f"Saved -> {OUT_DIR}/1_training_loss.png")

# ── Sample reconstructions ────────────────────────────────────
# Show 8 letters: original on top, reconstruction below
fig = plt.figure(figsize=(20, 6))
fig.patch.set_facecolor("#0f0f1a")
fig.suptitle("Original vs Reconstructed — Sample Test Images", color="white", fontsize=16, fontweight="bold")

show_classes = list(range(0, min(8, num_classes)))
gs = gridspec.GridSpec(2, len(show_classes), figure=fig, hspace=0.05, wspace=0.05)

for col, cls_idx in enumerate(show_classes):
    # Pick first test image of this class
    img_idx = (all_labels == cls_idx).nonzero()[0]
    if len(img_idx) == 0:
        continue
    img_idx = img_idx[0]

    orig = all_originals[img_idx, 0].numpy()
    recon = all_recons[img_idx, 0].numpy()

    for row, (img_data, row_label) in enumerate([(orig, "Original"), (recon, "Reconstructed")]):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img_data, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")
        if row == 0:
            ax.set_title(class_names[cls_idx], color="white", fontsize=12, fontweight="bold")
        if col == 0:
            ax.text(-3, 14, row_label, ha="right", va="center",
                    color="#e040fb" if row == 0 else "#00e5ff",
                    fontsize=10, fontweight="bold", rotation=0,
                    transform=ax.transData)

plt.savefig(f"{OUT_DIR}/2_reconstructions.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print(f"Saved -> {OUT_DIR}/2_reconstructions.png")

# ── Reconstruction error distribution ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for ax in axes:
    ax.set_facecolor("#0f0f1a")
fig.patch.set_facecolor("#0f0f1a")

# Left: histogram of clean vs anomaly errors
axes[0].hist(all_errors, bins=60, color="#e040fb", alpha=0.75, label="Clean (test set)", density=True)
axes[0].hist(anomaly_errors, bins=40, color="#ff5252", alpha=0.75, label="Corrupted (anomalies)", density=True)
axes[0].axvline(threshold, color="#ffd93d", linewidth=2, linestyle="--",
                label=f"Threshold ({ANOMALY_THRESHOLD_PERCENTILE}th pct) = {threshold:.4f}")
axes[0].set_title("Reconstruction Error Distribution", color="white", fontsize=13, fontweight="bold")
axes[0].set_xlabel("MSE Reconstruction Error", color="#aaaacc")
axes[0].set_ylabel("Density", color="#aaaacc")
axes[0].tick_params(colors="white")
axes[0].grid(color="#222244", linewidth=0.5)
for spine in axes[0].spines.values():
    spine.set_edgecolor("#333355")
axes[0].legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white")

# Right: mean reconstruction error per class
mean_error_per_class = [
    all_errors[all_labels == cls_idx].mean() if (all_labels == cls_idx).any() else 0
    for cls_idx in range(num_classes)
]
colors = plt.cm.plasma(np.linspace(0.15, 0.95, num_classes))
bars = axes[1].bar(class_names, mean_error_per_class, color=colors, edgecolor="none", width=0.7)
axes[1].set_title("Mean Reconstruction Error per Class", color="white", fontsize=13, fontweight="bold")
axes[1].set_xlabel("ASL Letter", color="#aaaacc")
axes[1].set_ylabel("Mean MSE", color="#aaaacc")
axes[1].tick_params(colors="white")
axes[1].grid(axis="y", color="#222244", linewidth=0.5)
for spine in axes[1].spines.values():
    spine.set_edgecolor("#333355")

fig.suptitle("Anomaly Detection — Reconstruction Error Analysis", color="white", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/3_anomaly_detection.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print(f"Saved -> {OUT_DIR}/3_anomaly_detection.png")

# ── Best and worst reconstructions ───────────────────────────
fig = plt.figure(figsize=(20, 8))
fig.patch.set_facecolor("#0f0f1a")
fig.suptitle("Easiest vs Hardest Images to Reconstruct", color="white", fontsize=15, fontweight="bold")
gs = gridspec.GridSpec(2, 8, figure=fig, hspace=0.3, wspace=0.05)

sorted_indices = np.argsort(all_errors)
best_indices = sorted_indices[:8]       # lowest reconstruction error
worst_indices = sorted_indices[-8:][::-1]  # highest reconstruction error

for col, (b_idx, w_idx) in enumerate(zip(best_indices, worst_indices)):
    for row, (idx, title_color, label) in enumerate([
        (b_idx, "#00e5ff", "Easy"),
        (w_idx, "#ff5252", "Hard")
    ]):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(all_originals[idx, 0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")
        ax.set_title(
            f"{class_names[all_labels[idx]]}\n{all_errors[idx]:.4f}",
            color=title_color, fontsize=8
        )
        if col == 0:
            ax.text(-3, 14, label, ha="right", va="center",
                    color=title_color, fontsize=10, fontweight="bold",
                    transform=ax.transData)

plt.savefig(f"{OUT_DIR}/4_best_worst_reconstructions.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print(f"Saved -> {OUT_DIR}/4_best_worst_reconstructions.png")

# ── Corrupted image examples ──────────────────────────────────
fig = plt.figure(figsize=(20, 9))
fig.patch.set_facecolor("#0f0f1a")
fig.suptitle("Anomaly Detection — Corrupted Image Examples\n(Red border = flagged as anomaly)",
             color="white", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(3, 8, figure=fig, hspace=0.35, wspace=0.05)

show_n = 8
for col in range(show_n):
    idx = col
    orig = all_originals[anomaly_indices[idx], 0].numpy()
    corr = corrupted_imgs[idx, 0].numpy()
    rec = corrupted_recon[idx, 0].numpy()
    is_flagged = anomaly_errors[idx] > threshold

    for row, (img_data, row_color, row_label) in enumerate([
        (orig,  "white",   "Original"),
        (corr,  "#ffd93d", "Corrupted"),
        (rec,   "#ff5252" if is_flagged else "#00e5ff", "Reconstructed"),
    ]):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img_data, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")
        if row == 2:
            label_str = f"ERR={anomaly_errors[idx]:.4f}\n{'⚠ FLAGGED' if is_flagged else 'not flagged'}"
            ax.set_title(label_str, color="#ff5252" if is_flagged else "#aaaacc", fontsize=7)
        if col == 0:
            ax.text(-3, 14, row_label, ha="right", va="center",
                    color=row_color, fontsize=9, fontweight="bold",
                    transform=ax.transData)

plt.savefig(f"{OUT_DIR}/5_corrupted_examples.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print(f"Saved -> {OUT_DIR}/5_corrupted_examples.png")


# =============================================================
# 7. SUMMARY
# =============================================================
summary = {
    "Model": "ConvAutoencoder",
    "Latent Dimension": LATENT_DIM,
    "Parameters": n_params,
    "Best Val Loss (MSE)": round(float(best_val_loss), 6),
    "Test Mean Reconstruction Error": round(float(all_errors.mean()), 6),
    "Test Std Reconstruction Error": round(float(all_errors.std()), 6),
    "Anomaly Threshold (percentile)": ANOMALY_THRESHOLD_PERCENTILE,
    "Anomaly Threshold (value)": round(float(threshold), 6),
    "Clean Images Flagged (%)": round(100 * flagged_clean / len(all_errors), 2),
    "Corrupted Images Flagged (%)": round(100 * flagged_anomalies / n_anomalies, 2),
    "Batch Size": BATCH_SIZE,
    "Learning Rate": LR,
    "Epochs": EPOCHS,
    "Patience": PATIENCE,
}

pd.DataFrame([summary]).to_csv(f"{OUT_DIR}/6_summary.csv", index=False)
with open(f"{OUT_DIR}/6_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 70)
print("5 · FINAL SUMMARY")
print("=" * 70)
for key, val in summary.items():
    print(f"  {key:<40}: {val}")
print(f"\nAll outputs saved in ./{OUT_DIR}/")
print("Done.")
