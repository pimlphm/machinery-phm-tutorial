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
def create_data_loaders(base_path="/content/turbofan_data", batch_size=64, window=32, stride=16):
    from sklearn.preprocessing import StandardScaler

    # Load raw data
    train_df, test_df = load_cmapss(base_path)

    # Create windows
    X_train, y_train = create_windows(train_df, window=window, stride=stride)
    X_test, y_test = create_windows(test_df, window=window, stride=stride)

    # === (3.1) Normalize input features per channel (sensor) ===
    B, T, F = X_train.shape
    scalers = []
    
    # Normalize each sensor channel separately
    for channel in range(F):
        # Extract all data for this channel across all samples and time steps
        train_channel_data = X_train[:, :, channel].reshape(-1, 1)
        test_channel_data = X_test[:, :, channel].reshape(-1, 1)
        
        # Fit scaler on training data for this channel
        scaler = StandardScaler()
        train_channel_scaled = scaler.fit_transform(train_channel_data)
        test_channel_scaled = scaler.transform(test_channel_data)
        
        # Reshape back and assign
        X_train[:, :, channel] = train_channel_scaled.reshape(B, T)
        X_test[:, :, channel] = test_channel_scaled.reshape(X_test.shape[0], T)
        
        scalers.append(scaler)

    # === (3.2) Keep original RUL labels (no normalization) ===
    # y_train and y_test remain unchanged

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

    return train_loader, val_loader, test_loader, scalers  # return per-channel scalers
