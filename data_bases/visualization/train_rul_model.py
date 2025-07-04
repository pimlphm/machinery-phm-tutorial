# === Core Python ===
import os
import numpy as np

# === PyTorch ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# === Optimizer & Scheduler ===
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# === Utility ===
from tqdm import tqdm  # Progress bar for training loops
def compute_reconstruction_loss(x_pred, x_target, mask):
    """
    Custom reconstruction loss:
    - Mean squared error over valid (non-padded) time steps and features
    - Applies sequence mask to exclude padding from the loss
    """
    mse = nn.MSELoss(reduction='none')  # No reduction so we can apply the mask
    raw_loss = mse(x_pred, x_target)   # Shape: [B, T, C]
    masked_loss = raw_loss * mask.unsqueeze(-1)  # Apply mask: [B, T, 1]
    return masked_loss.sum() / mask.sum()        # Normalize by valid element count
def train_rul_model_with_optional_reconstruction(
    model,
    train_loader,
    val_loader,
    compute_reconstruction_loss,  # Function to compute reconstruction loss
    num_epochs=50,
    patience=10,
    lr=1e-3,
    reconstruction_weight=2.0,  # Set to 0.0 to disable
    rul_mse_weight=3.0,
    rul_mae_weight=1.0,
    save_path="best_model.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    mse = nn.MSELoss(reduction='none')
    mae = nn.L1Loss(reduction='none')

    best_score = float("inf")
    wait = 0

    print("🚀 Starting training...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"[Epoch {epoch:02d}]"):
            x = batch['x'].to(device)             # [B, T, C]
            rul = batch['rul'].to(device)         # [B, T]
            mask = batch['mask'].to(device)       # [B, T]

            # Sequence slicing: input at t, predict t+1
            x_in = x[:, :-1]
            x_tgt = x[:, 1:]
            rul_tgt = rul[:, 1:]
            mask_tgt = mask[:, 1:]

            optimizer.zero_grad()

            x_pred, rul_pred = model(x_in)

            # === Losses ===
            total_loss = 0.0

            # Optional reconstruction loss
            if reconstruction_weight > 0:
                rec_loss = compute_reconstruction_loss(x_pred, x_tgt, mask_tgt)
                total_loss += reconstruction_weight * rec_loss
            else:
                rec_loss = torch.tensor(0.0, device=device)

            # RUL prediction losses
            rul_mse_raw = mse(rul_pred, rul_tgt)
            rul_mae_raw = mae(rul_pred, rul_tgt)

            rul_mse_loss = (rul_mse_raw * mask_tgt).sum() / mask_tgt.sum()
            rul_mae_loss = (rul_mae_raw * mask_tgt).sum() / mask_tgt.sum()

            total_loss += rul_mse_weight * rul_mse_loss + rul_mae_weight * rul_mae_loss

            # Backpropagation
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += total_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # === Validation ===
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                rul = batch['rul'].to(device)
                mask = batch['mask'].to(device)

                x_in = x[:, :-1]
                x_tgt = x[:, 1:]
                rul_tgt = rul[:, 1:]
                mask_tgt = mask[:, 1:]

                x_pred, rul_pred = model(x_in)

                rec_loss = compute_reconstruction_loss(x_pred, x_tgt, mask_tgt) if reconstruction_weight > 0 else 0.0

                rul_mse_raw = mse(rul_pred, rul_tgt)
                rul_mae_raw = mae(rul_pred, rul_tgt)

                rul_mse_loss = (rul_mse_raw * mask_tgt).sum() / mask_tgt.sum()
                rul_mae_loss = (rul_mae_raw * mask_tgt).sum() / mask_tgt.sum()

                val_loss = reconstruction_weight * rec_loss + rul_mse_weight * rul_mse_loss + rul_mae_weight * rul_mae_loss
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # === Early Stopping ===
        if avg_val_loss < best_score:
            best_score = avg_val_loss
            wait = 0
            torch.save(model.state_dict(), save_path)
            print("✅ Saved best model.")
        else:
            wait += 1
            if wait >= patience:
                print("⏹️ Early stopping triggered.")
                break
