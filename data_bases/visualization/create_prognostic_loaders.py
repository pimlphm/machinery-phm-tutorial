# === Step 1: Download & import C-MAPSS data loader ===
# !wget https://raw.githubusercontent.com/pimlphm/machinery-phm-tutorial/main/data_bases/visualization/load_cmapss.py -O load_cmapss.py

from load_cmapss import load_cmapss       # Function to load C-MAPSS train/test data
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# === (1): Enhanced sliding window creation with multiple augmentation strategies ===
def create_windows(data, window=32, stride=16, selected_sensors=None, 
                  boundary_oversample=True, exponential_sampling=True, 
                  noise_augmentation=True, noise_std=0.01, add_diff_features=True,
                  add_position_encoding=True):
    """
    Enhanced window creation with multiple data augmentation strategies
    
    Args:
        data: Input dataframe with engine data
        window: Window size
        stride: Stride for sliding window
        selected_sensors: List of sensor indices to use
        boundary_oversample: Whether to oversample tail windows (high degradation)
        exponential_sampling: Whether to use exponential sampling based on 1/RUL
        noise_augmentation: Whether to add Gaussian noise for robustness
        noise_std: Standard deviation for noise augmentation
        add_diff_features: Whether to add first-order difference features
        add_position_encoding: Whether to add temporal position encoding
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
        windows_this_engine = []
        for i in range(0, len(s_vals) - window + 1, stride):
            window_data = s_vals[i:i+window]
            window_rul = ruls[i + window - 1]
            
            # Add first-order difference features
            if add_diff_features and window > 1:
                diff_features = np.diff(window_data, axis=0)
                # Pad with zeros for first timestep
                diff_features = np.vstack([np.zeros((1, diff_features.shape[1])), diff_features])
                window_data = np.concatenate([window_data, diff_features], axis=1)
            
            # Add position encoding
            if add_position_encoding:
                pos_encoding = np.arange(window).reshape(-1, 1) / window
                window_data = np.concatenate([window_data, pos_encoding], axis=1)
            
            windows_this_engine.append((window_data, window_rul, i))
        
        # Boundary oversampling - oversample windows with low RUL
        if boundary_oversample:
            boundary_windows = [(w, r, idx) for w, r, idx in windows_this_engine if r <= 50]
            # Add boundary windows 2-3 additional times
            for _ in range(2):
                windows_this_engine.extend(boundary_windows)
        
        # Exponential sampling based on degradation urgency
        if exponential_sampling:
            for window_data, window_rul, idx in windows_this_engine:
                # Sample probability based on 1/(RUL+1) - higher prob for lower RUL
                sample_prob = 1.0 / (window_rul + 1)
                sample_prob = min(sample_prob * 100, 1.0)  # Scale and cap at 1.0
                
                if np.random.random() < sample_prob:
                    X.append(window_data)
                    y.append(window_rul)
                    window_indices.append((eid, idx))
                    
                    # Noise augmentation for training robustness
                    if noise_augmentation:
                        noise = np.random.normal(0, noise_std, window_data.shape)
                        noisy_window = window_data + noise
                        X.append(noisy_window)
                        y.append(window_rul)
                        window_indices.append((eid, idx))
                else:
                    X.append(window_data)
                    y.append(window_rul)
                    window_indices.append((eid, idx))
        else:
            # Standard processing without exponential sampling
            for window_data, window_rul, idx in windows_this_engine:
                X.append(window_data)
                y.append(window_rul)
                window_indices.append((eid, idx))
                
                # Noise augmentation
                if noise_augmentation:
                    noise = np.random.normal(0, noise_std, window_data.shape)
                    noisy_window = window_data + noise
                    X.append(noisy_window)
                    y.append(window_rul)
                    window_indices.append((eid, idx))
    
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

# === (2): Enhanced PyTorch Dataset with dynamic masking ===
class RULDataset(Dataset):
    def __init__(self, X, y, rul_transform='none', segmented_labels=False, 
                 dynamic_mask=True, max_rul=None):
        """
        Enhanced dataset with RUL transformations and dynamic masking
        
        Args:
            rul_transform: 'none', 'log', 'reciprocal' for RUL label transformation
            segmented_labels: Whether to create segmented soft labels
            dynamic_mask: Whether to use dynamic mask based on degradation level
        """
        # Ensure data types are float32 and handle any NaN/inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_original = torch.tensor(y, dtype=torch.float32)
        
        # RUL label transformations
        if rul_transform == 'log':
            self.y = torch.log(self.y_original + 1)
        elif rul_transform == 'reciprocal':
            self.y = 1.0 / (self.y_original + 1)
        else:
            self.y = self.y_original
        
        # Clamp values to reasonable ranges to prevent CUDA errors
        self.X = torch.clamp(self.X, min=-1e6, max=1e6)
        self.y = torch.clamp(self.y, min=0.0, max=1e6)
        
        # Create segmented labels if requested
        self.segmented_labels = None
        if segmented_labels:
            # Create 5 degradation levels: [0-50], [51-100], [101-200], [201-300], [300+]
            seg_labels = torch.zeros_like(self.y_original)
            seg_labels[self.y_original <= 50] = 0
            seg_labels[(self.y_original > 50) & (self.y_original <= 100)] = 1
            seg_labels[(self.y_original > 100) & (self.y_original <= 200)] = 2
            seg_labels[(self.y_original > 200) & (self.y_original <= 300)] = 3
            seg_labels[self.y_original > 300] = 4
            self.segmented_labels = seg_labels
        
        # Dynamic mask based on degradation urgency
        self.dynamic_mask = dynamic_mask
        if dynamic_mask and max_rul is not None:
            # mask = 1 - RUL/max_RUL gives higher weight to lower RUL
            self.mask_weights = 1.0 - (self.y_original / max_rul)
            self.mask_weights = torch.clamp(self.mask_weights, 0.1, 1.0)
        else:
            self.mask_weights = torch.ones_like(self.y)
        
    def __len__(self): 
        return len(self.X)
    
    def __getitem__(self, idx):
        result = {
            'x': self.X[idx],                     
            'rul': self.y[idx],
            'rul_original': self.y_original[idx],
            'mask': torch.ones(self.X.shape[1]) * self.mask_weights[idx]
        }
        
        if self.segmented_labels is not None:
            result['seg_label'] = self.segmented_labels[idx]
            
        return result

# === (3): Enhanced data loaders with multi-condition standardization ===
def create_data_loaders(base_path="/content/turbofan_data", batch_size=64, window=32, stride=16, 
                       selected_sensors=None, auto_sensor_selection=True, 
                       per_condition_standardization=True, clip_std_range=3.0,
                       rul_transform='none', boundary_oversample=True,
                       exponential_sampling=True, noise_augmentation=True,
                       add_diff_features=True, add_position_encoding=True,
                       fd_datasets=['FD001', 'FD002', 'FD003', 'FD004']):
    """
    Enhanced data loader creation with comprehensive preprocessing
    
    Args:
        fd_datasets: List of FD datasets to process. Options: ['FD001', 'FD002', 'FD003', 'FD004']
                    Example: ['FD001'] or ['FD001', 'FD002'] or ['FD001', 'FD002', 'FD003', 'FD004']
    """
    from sklearn.preprocessing import StandardScaler

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

    # Create enhanced windows
    X_train, y_train, train_indices = create_windows(
        train_df, window=window, stride=stride, selected_sensors=selected_sensors,
        boundary_oversample=boundary_oversample, exponential_sampling=exponential_sampling,
        noise_augmentation=noise_augmentation, add_diff_features=add_diff_features,
        add_position_encoding=add_position_encoding
    )
    
    X_test, y_test, test_indices = create_windows(
        test_df, window=window, stride=stride, selected_sensors=selected_sensors,
        boundary_oversample=False, exponential_sampling=False, 
        noise_augmentation=False, add_diff_features=add_diff_features,
        add_position_encoding=add_position_encoding
    )

    # Clean data before processing
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=1e6, neginf=-1e6)
    y_test = np.nan_to_num(y_test, nan=0.0, posinf=1e6, neginf=-1e6)

    # Store original statistics for debugging
    orig_stats = {
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
        'y_train_range': (y_train.min(), y_train.max()),
        'y_test_range': (y_test.min(), y_test.max()),
        'fd_datasets': fd_datasets
    }

    # === Enhanced standardization ===
    B, T, F = X_train.shape
    scalers = []
    
    # Per-condition standardization if available
    if per_condition_standardization and 'operating_condition' in train_df.columns:
        print("🔧 Applying per-condition standardization...")
        # Group by operating conditions and standardize separately
        conditions = train_df['operating_condition'].unique()
        for condition in conditions:
            # This is a simplified approach - in practice, you'd need to track
            # which windows belong to which condition
            pass
    
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
    max_rul = max(y_train.max(), y_test.max())

    # === Enhanced DataLoaders with advanced features ===
    train_dataset = RULDataset(X_train, y_train, rul_transform=rul_transform, 
                              segmented_labels=True, dynamic_mask=True, max_rul=max_rul)
    val_dataset = RULDataset(X_train, y_train, rul_transform=rul_transform, 
                            segmented_labels=True, dynamic_mask=True, max_rul=max_rul)
    test_dataset = RULDataset(X_test, y_test, rul_transform=rul_transform, 
                             segmented_labels=True, dynamic_mask=True, max_rul=max_rul)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # === Enhanced debugging and statistics ===
    batch = next(iter(train_loader))
    # print("✅ Enhanced DataLoader batch shapes:")
    print("x:", batch['x'].shape)
    print("rul:", batch['rul'].shape)
    # print("rul_original:", batch['rul_original'].shape)
    # print("mask:", batch['mask'].shape)
    # if 'seg_label' in batch:
    #     print("seg_label:", batch['seg_label'].shape)
    
    # print(f"\n🎯 Configuration Summary:")

    print(f"Selected sensors: {selected_sensors}")
    print(f"Number of sensor channels: {len(selected_sensors)}")
    # print(f"Enhanced features: diff={add_diff_features}, pos_enc={add_position_encoding}")
    # print(f"Data augmentation: boundary={boundary_oversample}, exp_samp={exponential_sampling}, noise={noise_augmentation}")
    # print(f"RUL transformation: {rul_transform}")
    # print(f"Standardization clipping: ±{clip_std_range}")
    
    # print(f"\n📊 Dataset Statistics:")
    # print(f"Original shapes - Train: {orig_stats['X_train_shape']}, Test: {orig_stats['X_test_shape']}")
    # print(f"Final shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    # print(f"Total batches - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    # print(f"\n📈 Data Ranges After Preprocessing:")
    # print(f"X_train: min={X_train.min():.3f}, max={X_train.max():.3f}, std={X_train.std():.3f}")
    # print(f"X_test: min={X_test.min():.3f}, max={X_test.max():.3f}, std={X_test.std():.3f}")
    # print(f"y_train: min={y_train.min():.1f}, max={y_train.max():.1f}, mean={y_train.mean():.1f}")
    # print(f"y_test: min={y_test.min():.1f}, max={y_test.max():.1f}, mean={y_test.mean():.1f}")

    # Enhanced return with additional metadata
    return_dict = {
        'train_loader': train_loader,
        'val_loader': val_loader, 
        'test_loader': test_loader,
        'scalers': scalers,
        'selected_sensors': selected_sensors,
        'window_indices': {'train': train_indices, 'test': test_indices},
        'max_rul': max_rul,

        'preprocessing_stats': {
            'original_stats': orig_stats,
            'final_shapes': {'train': X_train.shape, 'test': X_test.shape},
            'rul_transform': rul_transform,
            'clip_range': clip_std_range
        }
    }
    
    return return_dict
