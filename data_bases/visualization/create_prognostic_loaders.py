# === Step 1: Download & import C-MAPSS data loader ===
# Load helper to read NASA engine data (if not yet installed)
# !wget https://raw.githubusercontent.com/pimlphm/machinery-phm-tutorial/main/data_bases/visualization/load_cmapss.py -O load_cmapss.py

from load_cmapss import load_cmapss       # Function to load C-MAPSS train/test data
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader

# === (1): Slice engine data into sliding windows ===
# Each window = a short time segment + its RUL label
def create_windows(data, window=32, stride=16):
    X, y = [], []
    sensors = [f'sensor_{i}' for i in range(1, 22)]  # 21 sensor columns
    for eid in data['engine_id'].unique():
        series = data[data['engine_id'] == eid]
        s_vals, ruls = series[sensors].values, series['RUL'].values
        for i in range(0, len(s_vals) - window + 1, stride):
            X.append(s_vals[i:i+window])
            y.append(ruls[i + window - 1])
    return np.array(X), np.array(y)

# === (3): Wrap into PyTorch Dataset ===
# This formats each sample into a dictionary with input, label, and a valid-time mask
class RULDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return {
            'x': self.X[idx],                     # [window, features]
            'rul': self.y[idx],                   # single RUL value for the window endpoint
            'mask': torch.ones(self.X.shape[1])   # all time steps are valid
        }

def create_data_loaders(base_path="/content/turbofan_data", batch_size=64):
    # === (2): Load data & apply window slicing ===
    train_df, test_df = load_cmapss(base_path)                 # Load raw data
    X_train, y_train = create_windows(train_df)                # Slice train set
    X_test, y_test = create_windows(test_df)                   # Slice test set

    # === (4): Create DataLoaders for training/testing ===
    # Dataloaders split the dataset into batches and optionally shuffle
    train_loader = DataLoader(RULDataset(X_train, y_train), batch_size, shuffle=True)   # Train with shuffle
    val_loader   = DataLoader(RULDataset(X_train, y_train), batch_size, shuffle=False)  # Eval (no shuffle)
    test_loader  = DataLoader(RULDataset(X_test, y_test), batch_size, shuffle=False)    # Final test
    #shuffle=True randomly shuffles training data each epoch to improve generalization, while validation and test sets stay in order to ensure consistent and reproducible evaluation.

    # Get one batch from train_loader
    # A batch is a group of samples processed together before one weight update.
    # The number of batches per epoch = total samples ÷ batch size
    batch = next(iter(train_loader))

    print("✅ DataLoader batch shapes:")
    print("x:", batch['x'].shape)       # [batch_size, window, features]
    print("rul:", batch['rul'].shape)   # [batch_size] - single RUL value per window
    print("mask:", batch['mask'].shape) # [batch_size, window]

    # Print total number of batches in the loader
    print(f"\nTotal batches in train_loader: {len(train_loader)}")
    
    return train_loader, val_loader, test_loader
