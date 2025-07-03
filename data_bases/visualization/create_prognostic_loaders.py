# === Step 1: Download & import C-MAPSS data loader ===
# !wget https://raw.githubusercontent.com/pimlphm/machinery-phm-tutorial/main/data_bases/visualization/load_cmapss.py -O load_cmapss.py

from load_cmapss import load_cmapss
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# === (1): Slice engine data into sliding windows ===
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

# === (3): PyTorch Dataset wrapper ===
class RULDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return {
            'x': self.X[idx],                     # [window, features]
            'rul': self.y[idx],                   # scalar RUL
            'mask': torch.ones(self.X.shape[1])   # [window] - all valid steps
        }

# === (4): Loader creation + normalization ===
def create_data_loaders(base_path="/content/turbofan_data", batch_size=64):
    from sklearn.preprocessing import StandardScaler

    # Load raw C-MAPSS data
    train_df, test_df = load_cmapss(base_path)

    # Sliding window slicing
    X_train, y_train = create_windows(train_df)
    X_test, y_test = create_windows(test_df)

    # Flatten and fit StandardScaler on training data
    B, T, F = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, F)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(B, T, F)
    X_test_flat = X_test.reshape(-1, F)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

    # Normalize RUL to [0, 1]
    y_max = y_train.max()
    y_train_scaled = y_train / y_max
    y_test_scaled = y_test / y_max

    # Wrap in Datasets & Loaders
    train_loader = DataLoader(RULDataset(X_train_scaled, y_train_scaled), batch_size, shuffle=True)
    val_loader   = DataLoader(RULDataset(X_train_scaled, y_train_scaled), batch_size, shuffle=False)
    test_loader  = DataLoader(RULDataset(X_test_scaled, y_test_scaled), batch_size, shuffle=False)

    # Show example batch
    batch = next(iter(train_loader))
    print("✅ DataLoader batch shapes:")
    print("x:", batch['x'].shape)
    print("rul:", batch['rul'].shape)
    print("mask:", batch['mask'].shape)
    print(f"\nTotal batches in train_loader: {len(train_loader)}")
    print("ℹ️ Sensor data standardized. RUL normalized to [0, 1].")

    return train_loader, val_loader, test_loader, scaler, y_max
