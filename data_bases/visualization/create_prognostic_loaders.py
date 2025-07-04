# === Step 1: Download & import C-MAPSS data loader ===
# !wget https://raw.githubusercontent.com/pimlphm/machinery-phm-tutorial/main/data_bases/visualization/load_cmapss.py -O load_cmapss.py

from load_cmapss import load_cmapss       # Function to load C-MAPSS train/test data
import numpy as np, torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# === (1): Simple sliding window creation ===
def create_windows(data, window=32, stride=16, selected_sensors=None):
    """
    Simple window creation without augmentation
    
    Args:
        data: Input dataframe with engine data
        window: Window size
        stride: Stride for sliding window
        selected_sensors: List of sensor indices to use
    """
    X, y, window_indices = [], [], []
    sensors = [f'sensor_{i}' for i in range(1, 22)]
    
    # Select specific sensor channels if provided
    if selected_sensors is not None:
        sensors = [f'sensor_{i}' for i in selected_sensors]
    
    for eid in data['engine_id'].unique():
        series = data[data['engine_id'] == eid]
        s_vals, ruls = series[sensors].values, series['RUL'].values
        
        # Standard window creation
        for i in range(0, len(s_vals) - window + 1, stride):
            window_data = s_vals[i:i+window]
            window_rul = ruls[i + window - 1]
            
            X.append(window_data)
            y.append(window_rul)
            window_indices.append((eid, i))
    
    return np.array(X), np.array(y), window_indices

# === (1.1): Automatic sensor selection based on correlation ===
def auto_select_sensors(data, target_col='RUL', method='correlation', top_k=12):
    """
    Automatically select top-k sensors based on statistical criteria
    """
    sensors = [f'sensor_{i}' for i in range(1, 22)]
    
    if method == 'correlation':
        correlations = []
        for sensor in sensors:
            if sensor in data.columns:
                corr, _ = pearsonr(data[sensor], data[target_col])
                correlations.append((sensor, abs(corr)))
        
        # Sort by correlation and select top-k
        correlations.sort(key=lambda x: x[1], reverse=True)
        selected = [int(s[0].split('_')[1]) for s, _ in correlations[:top_k]]
        
    elif method == 'f_test':
        X = data[sensors].values
        y = data[target_col].values
        selector = SelectKBest(f_regression, k=top_k)
        selector.fit(X, y)
        selected_indices = selector.get_support(indices=True)
        selected = [i + 1 for i in selected_indices]  # Convert to 1-based indexing
    
    return selected

# === (2): Simple PyTorch Dataset ===
class RULDataset(Dataset):
    def __init__(self, X, y):
        """
        Simple dataset for RUL prediction
        """
        # Ensure data types are float32 and handle any NaN/inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
        # Clamp values to reasonable ranges to prevent CUDA errors
        self.X = torch.clamp(self.X, min=-1e6, max=1e6)
        self.y = torch.clamp(self.y, min=0.0, max=1e6)
        
    def __len__(self): 
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'x': self.X[idx],                     
            'rul': self.y[idx]
        }

# === (3): Simple data loaders with standardization ===
def create_data_loaders(base_path="/content/turbofan_data", batch_size=64, window=32, stride=16, 
                       selected_sensors=None, auto_sensor_selection=True, 
                       clip_std_range=3.0,
                       fd_datasets=['FD001', 'FD002', 'FD003', 'FD004']):
    """
    Simple data loader creation with basic preprocessing
    
    Args:
        fd_datasets: List of FD datasets to process. Options: ['FD001', 'FD002', 'FD003', 'FD004']
                    Example: ['FD001'] or ['FD001', 'FD002'] or ['FD001', 'FD002', 'FD003', 'FD004']
    """

    # Load raw data from specified FD datasets
    train_dfs, test_dfs = [], []
    
    for fd_name in fd_datasets:
        print(f"🔄 Loading {fd_name} dataset...")
        train_df, test_df = load_cmapss(base_path, dataset=fd_name)
        
        # Add dataset identifier for tracking
        train_df['dataset'] = fd_name
        test_df['dataset'] = fd_name
        
        train_dfs.append(train_df)
        test_dfs.append(test_df)
    
    # Combine all datasets
    if len(train_dfs) > 1:
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        print(f"📊 Combined {len(fd_datasets)} datasets: {fd_datasets}")
    else:
        train_df = train_dfs[0]
        test_df = test_dfs[0]
        print(f"📊 Using single dataset: {fd_datasets[0]}")
    
    # Automatic sensor selection if requested
    if auto_sensor_selection and selected_sensors is None:
        selected_sensors = auto_select_sensors(train_df, method='correlation', top_k=12)
        print(f"🤖 Auto-selected sensors: {selected_sensors}")
    elif selected_sensors is None:
        selected_sensors = [2, 3, 4, 7, 8, 9, 11, 12, 13, 15, 17, 20]

    # Create simple windows
    X_train, y_train, train_indices = create_windows(
        train_df, window=window, stride=stride, selected_sensors=selected_sensors
    )
    
    X_test, y_test, test_indices = create_windows(
        test_df, window=window, stride=stride, selected_sensors=selected_sensors
    )

    # Clean data before processing
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=1e6, neginf=-1e6)
    y_test = np.nan_to_num(y_test, nan=0.0, posinf=1e6, neginf=-1e6)

    # === Standardization ===
    B, T, F = X_train.shape
    scalers = []
    
    # Standardize each sensor channel separately
    for channel in range(F):
        # Extract all data for this sensor channel across all samples and time steps
        train_channel_data = X_train[:, :, channel].reshape(-1, 1)
        test_channel_data = X_test[:, :, channel].reshape(-1, 1)
        
        # Fit StandardScaler on training data for this specific sensor
        scaler = StandardScaler()
        train_channel_scaled = scaler.fit_transform(train_channel_data)
        test_channel_scaled = scaler.transform(test_channel_data)
        
        # Clip standardized values to prevent extreme outliers
        if clip_std_range > 0:
            train_channel_scaled = np.clip(train_channel_scaled, -clip_std_range, clip_std_range)
            test_channel_scaled = np.clip(test_channel_scaled, -clip_std_range, clip_std_range)
        
        # Clean scaled data
        train_channel_scaled = np.nan_to_num(train_channel_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
        test_channel_scaled = np.nan_to_num(test_channel_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Reshape back and assign normalized values
        X_train[:, :, channel] = train_channel_scaled.reshape(B, T)
        X_test[:, :, channel] = test_channel_scaled.reshape(X_test.shape[0], T)
        
        scalers.append(scaler)

    # RUL label processing - keep original ranges but ensure validity
    y_train = np.clip(y_train, 0.0, 1e6)
    y_test = np.clip(y_test, 0.0, 1e6)

    # === Simple DataLoaders ===
    train_dataset = RULDataset(X_train, y_train)
    val_dataset = RULDataset(X_train, y_train)
    test_dataset = RULDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # === Output information ===
    batch = next(iter(train_loader))
    print("x:", batch['x'].shape)
    print("rul:", batch['rul'].shape)
    print(f"Selected sensors: {selected_sensors}")
    print(f"Number of sensor channels: {len(selected_sensors)}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader, 
        'test_loader': test_loader,
        'scalers': scalers,
        'selected_sensors': selected_sensors
    }
