# === Step 1: Download & import C-MAPSS data loader ===
# !wget https://raw.githubusercontent.com/pimlphm/machinery-phm-tutorial/main/data_bases/visualization/load_cmapss.py -O load_cmapss.py

from load_cmapss import load_cmapss       # Function to load C-MAPSS train/test data
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# === (1): Slice engine data into sliding windows ===
def create_windows(data, window=32, stride=16):
    X, y = [], []
    sensors = [f'sensor_{i}' for i in range(1, 22)]
    for eid in data['engine_id'].unique():
        series = data[data['engine_id'] == eid]
        s_vals, ruls = series[sensors].values, series['RUL'].values
        for i in range(0, len(s_vals) - window + 1, stride):
            X.append(s_vals[i:i+window])
            y.append(ruls[i + window - 1])
    return np.array(X), np.array(y)

# === (2): PyTorch Dataset ===
class RULDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return {
            'x': self.X[idx],                     
            'rul': self.y[idx],                   
            'mask': torch.ones(self.X.shape[1])   
        }

# === (3): Create data loaders with normalization ===
def create_data_loaders(base_path="/content/turbofan_data", batch_size=64):
    from sklearn.preprocessing import StandardScaler

    # Load raw data
    train_df, test_df = load_cmapss(base_path)

    # Create windows
    X_train, y_train = create_windows(train_df)
    X_test, y_test = create_windows(test_df)

    # === (3.1) Normalize input features across channels ===
    B, T, F = X_train.shape
    X_train_flat = X_train.reshape(-1, F)
    X_test_flat = X_test.reshape(-1, F)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    X_train = X_train_scaled.reshape(B, T, F)
    X_test = X_test_scaled.reshape(X_test.shape[0], T, F)

    # === (3.2) Normalize RUL labels ===
    rul_max = y_train.max()
    y_train = y_train / rul_max
    y_test = y_test / rul_max

    # === (4): DataLoaders
    train_loader = DataLoader(RULDataset(X_train, y_train), batch_size, shuffle=True)
    val_loader   = DataLoader(RULDataset(X_train, y_train), batch_size, shuffle=False)
    test_loader  = DataLoader(RULDataset(X_test, y_test), batch_size, shuffle=False)

    # === (5): Print info
    batch = next(iter(train_loader))
    print("✅ DataLoader batch shapes:")
    print("x:", batch['x'].shape)
    print("rul:", batch['rul'].shape)
    print("mask:", batch['mask'].shape)
    print(f"\nTotal batches in train_loader: {len(train_loader)}")

    return train_loader, val_loader, test_loader, scaler, rul_max  # return scalers if needed
