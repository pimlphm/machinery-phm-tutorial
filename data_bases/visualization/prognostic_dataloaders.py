

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

# === Comprehensive CMAPSS Data Processing Pipeline ===
def process_cmapss_data(base_path="/content/turbofan_data", 
                       fd_datasets=['FD001', 'FD002', 'FD003', 'FD004'],
                       window_size=32, 
                       stride=16,
                       batch_size=64,
                       setting_normalization='kmeans',  # 'kmeans' or 'binning'
                       n_clusters_or_bins=5,
                       variance_threshold=1e-6,
                       selected_sensors=None,
                       engine_level_normalization=True):
    """
    Comprehensive CMAPSS data processing pipeline with improved preprocessing
    
    Args:
        base_path: Path to CMAPSS data files
        fd_datasets: List of FD datasets to process
        window_size: Size of sliding windows
        stride: Stride for sliding windows
        batch_size: Batch size for data loaders
        setting_normalization: Method for operational setting regularization ('kmeans' or 'binning')
        n_clusters_or_bins: Number of clusters/bins for setting regularization
        variance_threshold: Threshold for removing low-variance sensors
        selected_sensors: Specific sensors to use (if None, auto-select based on variance)
        engine_level_normalization: Whether to normalize at engine level vs global
    """
    
    print("🔄 Starting comprehensive CMAPSS data processing...")
    
    # === 1. Load raw data from specified FD datasets ===
    train_dfs, test_dfs = [], []
    
    for fd_name in fd_datasets:
        print(f"📂 Loading {fd_name} dataset...")
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
    
    # === 2. Operational Setting Regularization ===
    print(f"⚙️ Applying operational setting regularization using {setting_normalization}...")
    
    setting_cols = ['setting1', 'setting2', 'setting3']
    
    if setting_normalization == 'kmeans':
        # Use KMeans clustering to discretize operational settings
        kmeans = KMeans(n_clusters=n_clusters_or_bins, random_state=42)
        
        # Fit on training data
        train_settings = train_df[setting_cols].values
        setting_clusters = kmeans.fit_predict(train_settings)
        
        # Apply to test data
        test_settings = test_df[setting_cols].values
        test_clusters = kmeans.predict(test_settings)
        
        # Create one-hot encoded features
        for i in range(n_clusters_or_bins):
            train_df[f'setting_cluster_{i}'] = (setting_clusters == i).astype(int)
            test_df[f'setting_cluster_{i}'] = (test_clusters == i).astype(int)
        
        print(f"   ✅ Created {n_clusters_or_bins} operational setting clusters")
        
    elif setting_normalization == 'binning':
        # Use binning for each setting dimension
        for setting_col in setting_cols:
            # Calculate bins based on training data quantiles
            _, bins = pd.qcut(train_df[setting_col], q=n_clusters_or_bins, retbins=True, duplicates='drop')
            
            # Apply binning to both train and test
            train_df[f'{setting_col}_bin'] = pd.cut(train_df[setting_col], bins=bins, labels=False, include_lowest=True)
            test_df[f'{setting_col}_bin'] = pd.cut(test_df[setting_col], bins=bins, labels=False, include_lowest=True)
            
            # Create one-hot encoding
            for i in range(len(bins)-1):
                train_df[f'{setting_col}_bin_{i}'] = (train_df[f'{setting_col}_bin'] == i).astype(int)
                test_df[f'{setting_col}_bin_{i}'] = (test_df[f'{setting_col}_bin'] == i).astype(int)
        
        print(f"   ✅ Created binned features for operational settings")
    
    # === 3. Invalid Sensor Removal ===
    print("🔍 Identifying and removing invalid sensors...")
    
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    valid_sensors = []
    
    for sensor_col in sensor_cols:
        # Calculate variance across all engines and cycles
        sensor_variance = train_df[sensor_col].var()
        
        if sensor_variance > variance_threshold:
            valid_sensors.append(sensor_col)
        else:
            print(f"   ❌ Removing {sensor_col} (variance: {sensor_variance:.2e})")
    
    print(f"   ✅ Kept {len(valid_sensors)}/{len(sensor_cols)} sensors")
    
    # Update selected sensors if not provided
    if selected_sensors is None:
        # Convert to 1-based indexing for sensor numbers
        selected_sensors = [int(s.split('_')[1]) for s in valid_sensors]
    else:
        # Filter selected sensors to only include valid ones
        valid_sensor_numbers = [int(s.split('_')[1]) for s in valid_sensors]
        selected_sensors = [s for s in selected_sensors if s in valid_sensor_numbers]
    
    print(f"   📋 Final selected sensors: {selected_sensors}")
    
    # === 4. Engine-level Normalization ===
    print(f"📏 Applying {'engine-level' if engine_level_normalization else 'global'} sensor normalization...")
    
    sensor_cols_to_use = [f'sensor_{i}' for i in selected_sensors]
    scalers = {}
    
    if engine_level_normalization:
        # Normalize each sensor for each engine separately
        for sensor_col in sensor_cols_to_use:
            scalers[sensor_col] = {}
            
            # For each engine, fit a separate scaler
            for engine_id in train_df['engine_id'].unique():
                engine_data = train_df[train_df['engine_id'] == engine_id][sensor_col].values.reshape(-1, 1)
                
                if len(engine_data) > 1 and engine_data.std() > 1e-8:
                    scaler = StandardScaler()
                    scaler.fit(engine_data)
                    scalers[sensor_col][engine_id] = scaler
                    
                    # Apply to training data
                    train_df.loc[train_df['engine_id'] == engine_id, sensor_col] = scaler.transform(engine_data).flatten()
                else:
                    # Handle constant or near-constant sequences
                    scalers[sensor_col][engine_id] = None
            
            # For test data, use the closest training engine's scaler or global fallback
            global_scaler = StandardScaler()
            global_scaler.fit(train_df[sensor_col].values.reshape(-1, 1))
            
            for engine_id in test_df['engine_id'].unique():
                engine_data = test_df[test_df['engine_id'] == engine_id][sensor_col].values.reshape(-1, 1)
                
                # Try to find a corresponding training scaler or use global
                if engine_id in scalers[sensor_col] and scalers[sensor_col][engine_id] is not None:
                    scaler = scalers[sensor_col][engine_id]
                else:
                    scaler = global_scaler
                
                test_df.loc[test_df['engine_id'] == engine_id, sensor_col] = scaler.transform(engine_data).flatten()
        
        print("   ✅ Applied engine-level normalization")
        
    else:
        # Global normalization
        for sensor_col in sensor_cols_to_use:
            scaler = StandardScaler()
            
            # Fit on training data
            train_data = train_df[sensor_col].values.reshape(-1, 1)
            train_df[sensor_col] = scaler.fit_transform(train_data).flatten()
            
            # Apply to test data
            test_data = test_df[sensor_col].values.reshape(-1, 1)
            test_df[sensor_col] = scaler.transform(test_data).flatten()
            
            scalers[sensor_col] = scaler
        
        print("   ✅ Applied global normalization")
    
    # === 5. Sliding Window Creation ===
    print(f"🪟 Creating sliding windows (size: {window_size}, stride: {stride})...")
    
    def create_windows_improved(data, window_size, stride, sensor_indices):
        """Create sliding windows with improved preprocessing"""
        X, y, window_indices = [], [], []
        sensors = [f'sensor_{i}' for i in sensor_indices]
        
        for eid in data['engine_id'].unique():
            series = data[data['engine_id'] == eid].sort_values('cycle')
            s_vals = series[sensors].values
            ruls = series['RUL'].values
            
            # Create sliding windows
            for i in range(0, len(s_vals) - window_size + 1, stride):
                window_data = s_vals[i:i+window_size]
                window_rul = ruls[i + window_size - 1]  # RUL at the end of window
                
                # Clean data
                window_data = np.nan_to_num(window_data, nan=0.0, posinf=1e6, neginf=-1e6)
                window_rul = np.nan_to_num(window_rul, nan=0.0, posinf=1e6, neginf=-1e6)
                
                X.append(window_data)
                y.append(window_rul)
                window_indices.append((eid, i))
        
        return np.array(X), np.array(y), window_indices
    
    # Create windows for training and test data
    X_train, y_train, train_indices = create_windows_improved(
        train_df, window_size, stride, selected_sensors
    )
    
    X_test, y_test, test_indices = create_windows_improved(
        test_df, window_size, stride, selected_sensors
    )
    
    print(f"   ✅ Training windows: {X_train.shape}")
    print(f"   ✅ Test windows: {X_test.shape}")
    
    # === 6. Final Data Cleaning and Clipping ===
    print("🧹 Final data cleaning...")
    
    # Clip RUL values to reasonable ranges
    y_train = np.clip(y_train, 0.0, 1e6)
    y_test = np.clip(y_test, 0.0, 1e6)
    
    # Clamp window data to prevent CUDA errors
    X_train = np.clip(X_train, -1e6, 1e6)
    X_test = np.clip(X_test, -1e6, 1e6)
    
    # === 7. Create PyTorch Datasets and DataLoaders ===
    print("🔄 Creating PyTorch datasets and data loaders...")
    
    class RULDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
            
        def __len__(self): 
            return len(self.X)
        
        def __getitem__(self, idx):
            return {
                'x': self.X[idx],                     
                'rul': self.y[idx]
            }
    
    # Create datasets
    train_dataset = RULDataset(X_train, y_train)
    test_dataset = RULDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # Using train for validation
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # === 8. Summary Information ===
    print("\n📋 Processing Summary:")
    print(f"   📊 Datasets used: {fd_datasets}")
    print(f"   🔧 Selected sensors: {selected_sensors}")
    print(f"   📏 Window size: {window_size}, Stride: {stride}")
    print(f"   🏷️ Training samples: {len(train_dataset)}")
    print(f"   🧪 Test samples: {len(test_dataset)}")
    print(f"   ⚙️ Setting normalization: {setting_normalization}")
    print(f"   📐 Engine-level normalization: {engine_level_normalization}")
    
    # Show a sample batch
    sample_batch = next(iter(train_loader))
    print(f"   📦 Sample batch - X shape: {sample_batch['x'].shape}, y shape: {sample_batch['rul'].shape}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader, 
        'test_loader': test_loader,
        'scalers': scalers,
        'selected_sensors': selected_sensors,
        'train_df': train_df,
        'test_df': test_df,
        'processing_config': {
            'window_size': window_size,
            'stride': stride,
            'setting_normalization': setting_normalization,
            'n_clusters_or_bins': n_clusters_or_bins,
            'variance_threshold': variance_threshold,
            'engine_level_normalization': engine_level_normalization
        }
    }
