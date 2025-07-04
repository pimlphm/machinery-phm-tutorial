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
import pandas as pd


# === Load and Prepare CMAPSS Data ===
def load_cmapss(base_path, dataset=None):
    # Define column names for the CMAPSS data files (3 operational settings + 21 sensors)
    cols = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    # If a specific dataset is requested, process only that one
    if dataset is not None:
        # Extract dataset number from name (e.g., 'FD001' -> 1)
        dataset_num = int(dataset[-1])
        
        # Load training data for specified dataset
        train = pd.read_csv(f"{base_path}/train_{dataset}.txt", sep='\s+', header=None, names=cols)

        # Load test data and the corresponding RUL (remaining useful life) ground truth values
        test = pd.read_csv(f"{base_path}/test_{dataset}.txt", sep='\s+', header=None, names=cols)
        rul = pd.read_csv(f"{base_path}/RUL_{dataset}.txt", sep='\s+', header=None, names=['true_RUL'])

        # Calculate the RUL for training data by subtracting each cycle from its engine's max cycle
        max_cycle = train.groupby('engine_id')['cycle'].transform('max')  # Each engine's max cycle
        train['RUL'] = max_cycle - train['cycle']  # RUL = max_cycle - current_cycle
        train['dataset'] = dataset_num  # Label the subset ID

        # Compute max cycle per engine in test set to determine current age
        test['max_cycle'] = test.groupby('engine_id')['cycle'].transform('max')

        # Assign RUL to each time step of each test engine based on true_RUL at final cycle
        test['RUL'] = test.apply(
            lambda row: rul.iloc[int(row['engine_id']) - 1, 0] + (row['max_cycle'] - row['cycle']),
            axis=1
        )

        # Drop the temporary max_cycle column
        test.drop('max_cycle', axis=1, inplace=True)
        test['dataset'] = dataset_num  # Add subset label

        return train, test
    
    # Original behavior: load all datasets if no specific dataset is requested
    # Initialize empty DataFrames to hold combined training and test data from all four subdatasets (FD001–FD004)
    train_all, test_all = pd.DataFrame(), pd.DataFrame()

    # Iterate through all four CMAPSS subsets (FD001 to FD004)
    for i in range(1, 5):
        # Load training data for current subset
        train = pd.read_csv(f"{base_path}/train_FD00{i}.txt", sep='\s+', header=None, names=cols)

        # Load test data and the corresponding RUL (remaining useful life) ground truth values
        test = pd.read_csv(f"{base_path}/test_FD00{i}.txt", sep='\s+', header=None, names=cols)
        rul = pd.read_csv(f"{base_path}/RUL_FD00{i}.txt", sep='\s+', header=None, names=['true_RUL'])

        # Calculate the RUL for training data by subtracting each cycle from its engine's max cycle
        max_cycle = train.groupby('engine_id')['cycle'].transform('max')  # Each engine's max cycle
        train['RUL'] = max_cycle - train['cycle']  # RUL = max_cycle - current_cycle
        train['dataset'] = i  # Label the subset ID (1 to 4)
        train_all = pd.concat([train_all, train], ignore_index=True)  # Append to the global training set

        # Compute max cycle per engine in test set to determine current age
        test['max_cycle'] = test.groupby('engine_id')['cycle'].transform('max')

        # To prevent duplicate engine IDs across different subsets, shift test engine IDs by offset
        offset = test_all['engine_id'].max() + 1 if not test_all.empty else 0
        test['engine_id'] += offset  # Shift engine_id to ensure uniqueness

        # Assign RUL to each time step of each test engine based on true_RUL at final cycle
        test['RUL'] = test.apply(
            lambda row: rul.iloc[int(row['engine_id']) - 1 - offset, 0] + (row['max_cycle'] - row['cycle']),
            axis=1
        )

        # Drop the temporary max_cycle column
        test.drop('max_cycle', axis=1, inplace=True)
        test['dataset'] = i  # Add subset label
        test_all = pd.concat([test_all, test], ignore_index=True)  # Append to the global test set

    # Return concatenated training and test DataFrames covering FD001 to FD004
    return train_all, test_all
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
                       selected_sensors=None,
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
    
    # Use provided sensor selection or default sensors
    if selected_sensors is None:
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
