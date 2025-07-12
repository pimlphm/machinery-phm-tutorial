import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# === Visualization Function ===
def plot_condition_normalization_comparison(
    data_root: str,
    subset: str = 'FD002',
    sensor_indices: list = [1, 2, 3, 4],
    n_clusters: int = 6,
    figsize: tuple = (15, 10)
):
    """
    Plot comparison of sensor data before and after condition normalization
    
    Args:
        data_root: Path to CMAPSS data files
        subset: Dataset subset to analyze (e.g., 'FD002')
        sensor_indices: List of sensor indices to plot (1-21)
        n_clusters: Number of clusters for condition grouping
        figsize: Figure size for the plot
    """
    # Define column names
    columns = ['unit_number','time_in_cycles','setting_1','setting_2','setting_3'] \
            + [f'sensor_{i}' for i in range(1, 22)]
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    setting_cols = ['setting_1','setting_2','setting_3']
    
    # Load data
    file_path = os.path.join(data_root, f"train_{subset}.txt")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    df = pd.read_csv(file_path, sep='\s+', header=None, names=columns)
    
    # Group by operating conditions using KMeans clustering
    df['condition_id'] = KMeans(n_clusters=n_clusters, random_state=0) \
                           .fit_predict(df[setting_cols])
    
    # Create normalized version
    df_normalized = df.copy()
    for cond in df['condition_id'].unique():
        mask = df['condition_id'] == cond
        scaled_values = StandardScaler().fit_transform(df.loc[mask, sensor_cols])
        df_normalized.loc[mask, sensor_cols] = scaled_values.astype('float64')
    
    # Create subplots
    n_sensors = len(sensor_indices)
    fig, axes = plt.subplots(2, n_sensors, figsize=figsize)
    if n_sensors == 1:
        axes = axes.reshape(2, 1)
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    for i, sensor_idx in enumerate(sensor_indices):
        sensor_col = f'sensor_{sensor_idx}'
        
        # Plot original data
        ax_orig = axes[0, i]
        for cond in sorted(df['condition_id'].unique()):
            cond_data = df[df['condition_id'] == cond]
            ax_orig.scatter(cond_data['time_in_cycles'], cond_data[sensor_col], 
                          c=[colors[cond]], alpha=0.6, s=1, label=f'Condition {cond}')
        ax_orig.set_title(f'Original {sensor_col}')
        ax_orig.set_xlabel('Time Cycles')
        ax_orig.set_ylabel('Sensor Value')
        if i == 0:
            ax_orig.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot normalized data
        ax_norm = axes[1, i]
        for cond in sorted(df_normalized['condition_id'].unique()):
            cond_data = df_normalized[df_normalized['condition_id'] == cond]
            ax_norm.scatter(cond_data['time_in_cycles'], cond_data[sensor_col], 
                          c=[colors[cond]], alpha=0.6, s=1, label=f'Condition {cond}')
        ax_norm.set_title(f'Normalized {sensor_col}')
        ax_norm.set_xlabel('Time Cycles')
        ax_norm.set_ylabel('Normalized Value')
        if i == 0:
            ax_norm.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle(f'Condition Normalization Comparison - {subset}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nStatistics for {subset}:")
    print(f"Number of conditions identified: {n_clusters}")
    print(f"Condition distribution:")
    print(df['condition_id'].value_counts().sort_index())

# === Data Loader ===
def prepare_cmapss_loaders(
    data_root: str,
    subsets: list = ['FD001','FD002','FD003','FD004'],
    condition_normalize: list = ['FD002','FD004'],
    batch_size: int = 8,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
):
    """
    Step 1: Define column names and sensor columns for CMAPSS dataset
    This includes unit number, time cycles, 3 settings, and 21 sensors
    """
    columns = ['unit_number','time_in_cycles','setting_1','setting_2','setting_3'] \
            + [f'sensor_{i}' for i in range(1, 22)]
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    setting_cols = ['setting_1','setting_2','setting_3']

    all_samples = []
    for subset in subsets:
        file_path = os.path.join(data_root, f"train_{subset}.txt")
        if not os.path.exists(file_path):
            continue

        """
        Step 2: Load and parse the dataset file
        Each file contains space-separated values without headers
        """
        df = pd.read_csv(file_path, sep='\s+', header=None, names=columns)

        """
        Step 3: Calculate Remaining Useful Life (RUL) for each unit
        RUL = maximum cycle time - current cycle time for each unit
        """
        max_cycle = df.groupby('unit_number')['time_in_cycles'] \
                      .max().reset_index().rename(columns={'time_in_cycles':'max_cycle'})
        df = df.merge(max_cycle, on='unit_number')
        df['RUL'] = df['max_cycle'] - df['time_in_cycles']
        df.drop('max_cycle', axis=1, inplace=True)

        """
        Step 4: Normalize sensor data with proper dtype handling
        Fix pandas FutureWarning by ensuring compatible dtypes
        """
        if subset in condition_normalize:
            # Group by operating conditions using KMeans clustering
            df['condition_id'] = KMeans(n_clusters=6, random_state=0) \
                                   .fit_predict(df[setting_cols])
            df_scaled = df.copy()
            for cond in df['condition_id'].unique():
                mask = df['condition_id']==cond
                # Fix: Explicitly convert to float64 to match pandas dtype expectations
                scaled_values = StandardScaler().fit_transform(df.loc[mask, sensor_cols])
                df_scaled.loc[mask, sensor_cols] = scaled_values.astype('float64')
            df = df_scaled.drop('condition_id', axis=1)
        else:
            # Fix: Explicitly convert to float64 for standard normalization
            scaled_values = StandardScaler().fit_transform(df[sensor_cols])
            df[sensor_cols] = scaled_values.astype('float64')

        """
        Step 5: Extract sequences for each unit and sort by time
        Each unit becomes a separate time series sample
        """
        for uid in df['unit_number'].unique():
            seq_df = df[df['unit_number']==uid] \
                        .sort_values('time_in_cycles') \
                        .reset_index(drop=True)
            all_samples.append({'unit':int(uid),'subset':subset,'data':seq_df})

    """
    Step 6: Split data into train/validation/test sets
    Use deterministic shuffling for reproducibility
    """
    np.random.seed(0)
    np.random.shuffle(all_samples)
    N = len(all_samples)
    n1 = int(train_ratio * N)
    n2 = int((train_ratio+val_ratio) * N)
    train_samples = all_samples[:n1]
    val_samples   = all_samples[n1:n2]
    test_samples  = all_samples[n2:]

    """
    Step 7: Define PyTorch Dataset class for CMAPSS data
    Converts pandas dataframes to PyTorch tensors
    """
    class CMAPSSDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self): return len(self.samples)
        def __getitem__(self, idx):
            s = self.samples[idx]
            df_seq = s['data']
            x = torch.tensor(df_seq[sensor_cols].values, dtype=torch.float32)
            rul = torch.tensor(df_seq['RUL'].values, dtype=torch.float32)
            return {'x':x,'rul':rul,'unit':s['unit'],'subset':s['subset']}

    """
    Step 8: Define collate function for batching variable-length sequences
    Pads sequences to the same length and creates attention masks
    """
    def collate_fn(batch):
        lengths = [b['x'].size(0) for b in batch]
        T_max = max(lengths); C = batch[0]['x'].size(1); B = len(batch)
        x_batch = torch.zeros(B,T_max,C)
        rul_batch = torch.zeros(B,T_max)
        mask = torch.zeros(B,T_max)
        for i,b in enumerate(batch):
            L = b['x'].size(0)
            x_batch[i,:L] = b['x']
            rul_batch[i,:L] = b['rul']
            mask[i,:L] = 1
        return {'x':x_batch,'rul':rul_batch,'mask':mask}

    def make_loader(samples, shuffle):
        ds = CMAPSSDataset(samples)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return make_loader(train_samples, True), make_loader(val_samples, False), make_loader(test_samples, False)


# # === Basic Python ===
# import os

# # === Data Handling ===
# import pandas as pd
# import numpy as np

# # === Plotting (可选，如果你做可视化分析) ===
# import matplotlib.pyplot as plt
# import seaborn as sns

# # === Machine Learning Utilities ===
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.cluster import KMeans

# # === Import additional libraries ===
# import numpy as np, torch
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import SelectKBest, f_regression
# from scipy.stats import pearsonr
# from scipy import stats
# import warnings
# warnings.filterwarnings('ignore')


# # === Comprehensive CMAPSS Data Processing Pipeline ===
# def process_cmapss_data(base_path="/content/turbofan_data", 
#                        fd_datasets=['FD001', 'FD002', 'FD003', 'FD004'],
#                        window_size=32, 
#                        stride=16,
#                        batch_size=64,
#                        setting_normalization='kmeans',  # 'kmeans' or 'binning'
#                        n_clusters_or_bins=5,
#                        variance_threshold=1e-6,
#                        selected_sensors=None,
#                        engine_level_normalization=True):
#     """
#     Comprehensive CMAPSS data processing pipeline with improved preprocessing
    
#     Args:
#         base_path: Path to CMAPSS data files
#         fd_datasets: List of FD datasets to process
#         window_size: Size of sliding windows
#         stride: Stride for sliding windows
#         batch_size: Batch size for data loaders
#         setting_normalization: Method for operational setting regularization ('kmeans' or 'binning')
#         n_clusters_or_bins: Number of clusters/bins for setting regularization
#         variance_threshold: Threshold for removing low-variance sensors
#         selected_sensors: Specific sensors to use (if None, auto-select based on variance)
#         engine_level_normalization: Whether to normalize at engine level vs global
#     """
    
#     # === Integrated CMAPSS data loading function ===
#     def load_cmapss_internal(base_path, dataset=None):
#         # Define column names for the CMAPSS data files (3 operational settings + 21 sensors)
#         cols = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor_{i}' for i in range(1, 22)]
        
#         # If a specific dataset is requested, process only that one
#         if dataset is not None:
#             # Extract dataset number from name (e.g., 'FD001' -> 1)
#             dataset_num = int(dataset[-1])
            
#             # Load training data for specified dataset
#             train = pd.read_csv(f"{base_path}/train_{dataset}.txt", sep='\s+', header=None, names=cols)

#             # Load test data and the corresponding RUL (remaining useful life) ground truth values
#             test = pd.read_csv(f"{base_path}/test_{dataset}.txt", sep='\s+', header=None, names=cols)
#             rul = pd.read_csv(f"{base_path}/RUL_{dataset}.txt", sep='\s+', header=None, names=['true_RUL'])

#             # Calculate the RUL for training data by subtracting each cycle from its engine's max cycle
#             max_cycle = train.groupby('engine_id')['cycle'].transform('max')  # Each engine's max cycle
#             train['RUL'] = max_cycle - train['cycle']  # RUL = max_cycle - current_cycle
#             train['dataset'] = dataset_num  # Label the subset ID

#             # Compute max cycle per engine in test set to determine current age
#             test['max_cycle'] = test.groupby('engine_id')['cycle'].transform('max')

#             # Assign RUL to each time step of each test engine based on true_RUL at final cycle
#             test['RUL'] = test.apply(
#                 lambda row: rul.iloc[int(row['engine_id']) - 1, 0] + (row['max_cycle'] - row['cycle']),
#                 axis=1
#             )

#             # Drop the temporary max_cycle column
#             test.drop('max_cycle', axis=1, inplace=True)
#             test['dataset'] = dataset_num  # Add subset label

#             # Clip RUL values only for training data: remove bottom 5% and top 5% of RUL values
#             train_rul_quantiles = train['RUL'].quantile([0.05, 0.95])
#             train = train[(train['RUL'] >= train_rul_quantiles[0.05]) & (train['RUL'] <= train_rul_quantiles[0.95])]
            
#             # Keep test data unclipped for real evaluation
            
#             return train, test
        
#         # Original behavior: load all datasets if no specific dataset is requested
#         # Initialize empty DataFrames to hold combined training and test data from all four subdatasets (FD001–FD004)
#         train_all, test_all = pd.DataFrame(), pd.DataFrame()

#         # Iterate through all four CMAPSS subsets (FD001 to FD004)
#         for i in range(1, 5):
#             # Load training data for current subset
#             train = pd.read_csv(f"{base_path}/train_FD00{i}.txt", sep='\s+', header=None, names=cols)

#             # Load test data and the corresponding RUL (remaining useful life) ground truth values
#             test = pd.read_csv(f"{base_path}/test_FD00{i}.txt", sep='\s+', header=None, names=cols)
#             rul = pd.read_csv(f"{base_path}/RUL_FD00{i}.txt", sep='\s+', header=None, names=['true_RUL'])

#             # Calculate the RUL for training data by subtracting each cycle from its engine's max cycle
#             max_cycle = train.groupby('engine_id')['cycle'].transform('max')  # Each engine's max cycle
#             train['RUL'] = max_cycle - train['cycle']  # RUL = max_cycle - current_cycle
#             train['dataset'] = i  # Label the subset ID (1 to 4)
#             train_all = pd.concat([train_all, train], ignore_index=True)  # Append to the global training set

#             # Compute max cycle per engine in test set to determine current age
#             test['max_cycle'] = test.groupby('engine_id')['cycle'].transform('max')

#             # To prevent duplicate engine IDs across different subsets, shift test engine IDs by offset
#             offset = test_all['engine_id'].max() + 1 if not test_all.empty else 0
#             test['engine_id'] += offset  # Shift engine_id to ensure uniqueness

#             # Assign RUL to each time step of each test engine based on true_RUL at final cycle
#             test['RUL'] = test.apply(
#                 lambda row: rul.iloc[int(row['engine_id']) - 1 - offset, 0] + (row['max_cycle'] - row['cycle']),
#                 axis=1
#             )

#             # Drop the temporary max_cycle column
#             test.drop('max_cycle', axis=1, inplace=True)
#             test['dataset'] = i  # Add subset label
#             test_all = pd.concat([test_all, test], ignore_index=True)  # Append to the global test set

#         # Clip RUL values only for training data: remove bottom 5% and top 5% of RUL values
#         train_rul_quantiles = train_all['RUL'].quantile([0.05, 0.95])
#         train_all = train_all[(train_all['RUL'] >= train_rul_quantiles[0.05]) & (train_all['RUL'] <= train_rul_quantiles[0.95])]
        
#         # Keep test data unclipped for real evaluation

#         # Return concatenated training and test DataFrames covering FD001 to FD004
#         return train_all, test_all
    
#     # === 1. Load raw data from specified FD datasets ===
#     train_dfs, test_dfs = [], []
    
#     for fd_name in fd_datasets:
#         train_df, test_df = load_cmapss_internal(base_path, dataset=fd_name)
        
#         # Add dataset identifier for tracking
#         train_df['dataset'] = fd_name
#         test_df['dataset'] = fd_name
        
#         train_dfs.append(train_df)
#         test_dfs.append(test_df)
    
#     # Combine all datasets
#     if len(train_dfs) > 1:
#         train_df = pd.concat(train_dfs, ignore_index=True)
#         test_df = pd.concat(test_dfs, ignore_index=True)
#     else:
#         train_df = train_dfs[0]
#         test_df = test_dfs[0]
    
#     # === 2. Operational Setting Regularization ===
#     setting_cols = ['setting1', 'setting2', 'setting3']
    
#     if setting_normalization == 'kmeans':
#         # Use KMeans clustering to discretize operational settings
#         kmeans = KMeans(n_clusters=n_clusters_or_bins, random_state=42)
        
#         # Fit on training data
#         train_settings = train_df[setting_cols].values
#         setting_clusters = kmeans.fit_predict(train_settings)
        
#         # Apply to test data
#         test_settings = test_df[setting_cols].values
#         test_clusters = kmeans.predict(test_settings)
        
#         # Create one-hot encoded features
#         for i in range(n_clusters_or_bins):
#             train_df[f'setting_cluster_{i}'] = (setting_clusters == i).astype(int)
#             test_df[f'setting_cluster_{i}'] = (test_clusters == i).astype(int)
        
#     elif setting_normalization == 'binning':
#         # Use binning for each setting dimension
#         for setting_col in setting_cols:
#             # Calculate bins based on training data quantiles
#             _, bins = pd.qcut(train_df[setting_col], q=n_clusters_or_bins, retbins=True, duplicates='drop')
            
#             # Apply binning to both train and test
#             train_df[f'{setting_col}_bin'] = pd.cut(train_df[setting_col], bins=bins, labels=False, include_lowest=True)
#             test_df[f'{setting_col}_bin'] = pd.cut(test_df[setting_col], bins=bins, labels=False, include_lowest=True)
            
#             # Create one-hot encoding
#             for i in range(len(bins)-1):
#                 train_df[f'{setting_col}_bin_{i}'] = (train_df[f'{setting_col}_bin'] == i).astype(int)
#                 test_df[f'{setting_col}_bin_{i}'] = (test_df[f'{setting_col}_bin'] == i).astype(int)
    
#     # === 3. Invalid Sensor Removal ===
#     sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
#     valid_sensors = []
    
#     for sensor_col in sensor_cols:
#         # Calculate variance across all engines and cycles
#         sensor_variance = train_df[sensor_col].var()
        
#         if sensor_variance > variance_threshold:
#             valid_sensors.append(sensor_col)
    
#     # Update selected sensors if not provided
#     if selected_sensors is None:
#         # Convert to 1-based indexing for sensor numbers
#         selected_sensors = [int(s.split('_')[1]) for s in valid_sensors]
#     else:
#         # Filter selected sensors to only include valid ones
#         valid_sensor_numbers = [int(s.split('_')[1]) for s in valid_sensors]
#         selected_sensors = [s for s in selected_sensors if s in valid_sensor_numbers]
    
#     # === 4. Engine-level Normalization ===
#     sensor_cols_to_use = [f'sensor_{i}' for i in selected_sensors]
#     scalers = {}
    
#     if engine_level_normalization:
#         # Normalize each sensor for each engine separately
#         for sensor_col in sensor_cols_to_use:
#             scalers[sensor_col] = {}
            
#             # For each engine, fit a separate scaler
#             for engine_id in train_df['engine_id'].unique():
#                 engine_data = train_df[train_df['engine_id'] == engine_id][sensor_col].values.reshape(-1, 1)
                
#                 if len(engine_data) > 1 and engine_data.std() > 1e-8:
#                     scaler = StandardScaler()
#                     scaler.fit(engine_data)
#                     scalers[sensor_col][engine_id] = scaler
                    
#                     # Apply to training data
#                     train_df.loc[train_df['engine_id'] == engine_id, sensor_col] = scaler.transform(engine_data).flatten()
#                 else:
#                     # Handle constant or near-constant sequences
#                     scalers[sensor_col][engine_id] = None
            
#             # For test data, use the closest training engine's scaler or global fallback
#             global_scaler = StandardScaler()
#             global_scaler.fit(train_df[sensor_col].values.reshape(-1, 1))
            
#             for engine_id in test_df['engine_id'].unique():
#                 engine_data = test_df[test_df['engine_id'] == engine_id][sensor_col].values.reshape(-1, 1)
                
#                 # Try to find a corresponding training scaler or use global
#                 if engine_id in scalers[sensor_col] and scalers[sensor_col][engine_id] is not None:
#                     scaler = scalers[sensor_col][engine_id]
#                 else:
#                     scaler = global_scaler
                
#                 test_df.loc[test_df['engine_id'] == engine_id, sensor_col] = scaler.transform(engine_data).flatten()
        
#     else:
#         # Global normalization
#         for sensor_col in sensor_cols_to_use:
#             scaler = StandardScaler()
            
#             # Fit on training data
#             train_data = train_df[sensor_col].values.reshape(-1, 1)
#             train_df[sensor_col] = scaler.fit_transform(train_data).flatten()
            
#             # Apply to test data
#             test_data = test_df[sensor_col].values.reshape(-1, 1)
#             test_df[sensor_col] = scaler.transform(test_data).flatten()
            
#             scalers[sensor_col] = scaler
    
#     # === 5. Statistical Feature Extraction from Sliding Windows ===
#     def extract_statistical_features(window_data):
#         """
#         对滑窗内的每个通道在整个窗口T内做统计分析
#         输入: window_data - shape [T, F] (时间步，特征数)
#         输出: features - shape [F, 统计特征数] (每个传感器的统计特征向量)
#         """
#         T, F = window_data.shape
#         num_stats = 9  # 统计特征数量
#         features = np.zeros((F, num_stats))
        
#         for f in range(F):
#             channel_data = window_data[:, f]
            
#             # 基础统计特征
#             mean_val = np.mean(channel_data)                    # 均值 - 表示整体水平
#             std_val = np.std(channel_data)                      # 标准差 - 波动范围，反映状态稳定性
#             min_val = np.min(channel_data)                      # 最小值 - 捕捉尖峰或故障信号
#             max_val = np.max(channel_data)                      # 最大值 - 捕捉尖峰或故障信号
#             range_val = max_val - min_val                       # 变化幅度
#             median_val = np.median(channel_data)                # 中位数 - 抗异常值
#             q25 = np.percentile(channel_data, 25)               # 25分位数
#             q75 = np.percentile(channel_data, 75)               # 75分位数
            
#             # 线性趋势 - 表征退化方向
#             time_indices = np.arange(T)
#             if T > 1:
#                 slope, _, _, _, _ = stats.linregress(time_indices, channel_data)
#             else:
#                 slope = 0
            
#             # 存储为 [通道, 统计特征] 格式
#             features[f, :] = [mean_val, std_val, min_val, max_val, range_val, 
#                              median_val, q25, q75, slope]
        
#         return features
    
#     def create_windows_with_statistical_features(data, window_size, stride, sensor_indices):
#         """创建滑动窗口并提取统计特征，返回 [batch, 通道, 统计特征] 格式，保持时间顺序"""
#         X, y, window_indices = [], [], []
#         sensors = [f'sensor_{i}' for i in sensor_indices]
        
#         # 保持引擎ID的原始顺序，不打乱
#         for eid in sorted(data['engine_id'].unique()):
#             series = data[data['engine_id'] == eid].sort_values('cycle')
#             s_vals = series[sensors].values
#             ruls = series['RUL'].values
            
#             # Create sliding windows，保持时间顺序
#             for i in range(0, len(s_vals) - window_size + 1, stride):
#                 window_data = s_vals[i:i+window_size]
#                 window_rul = ruls[i + window_size - 1]  # RUL at the end of window
                
#                 # Clean data
#                 window_data = np.nan_to_num(window_data, nan=0.0, posinf=1e6, neginf=-1e6)
#                 window_rul = np.nan_to_num(window_rul, nan=0.0, posinf=1e6, neginf=-1e6)
                
#                 # 提取统计特征 - 从 [T, F] 到 [F, 统计特征数]
#                 statistical_features = extract_statistical_features(window_data)
                
#                 X.append(statistical_features)
#                 y.append(window_rul)
#                 window_indices.append((eid, i))
        
#         return np.array(X), np.array(y), window_indices
    
#     # Create windows with statistical features for training and test data
#     X_train, y_train, train_indices = create_windows_with_statistical_features(
#         train_df, window_size, stride, selected_sensors
#     )
    
#     X_test, y_test, test_indices = create_windows_with_statistical_features(
#         test_df, window_size, stride, selected_sensors
#     )
    
#     # === 6. Final Data Cleaning and Clipping ===
#     # Clip RUL values to reasonable ranges
#     y_train = np.clip(y_train, 0.0, 1e6)
#     y_test = np.clip(y_test, 0.0, 1e6)
    
#     # Clamp feature data to prevent CUDA errors
#     X_train = np.clip(X_train, -1e6, 1e6)
#     X_test = np.clip(X_test, -1e6, 1e6)
    
#     # === 7. Create PyTorch Datasets and DataLoaders ===
#     class RULDataset(Dataset):
#         def __init__(self, X, y):
#             self.X = torch.tensor(X, dtype=torch.float32)
#             self.y = torch.tensor(y, dtype=torch.float32)
            
#         def __len__(self): 
#             return len(self.X)
        
#         def __getitem__(self, idx):
#             # Return tuple instead of dictionary to avoid warnings
#             return self.X[idx], self.y[idx]
    
#     # Create datasets
#     train_dataset = RULDataset(X_train, y_train)
#     test_dataset = RULDataset(X_test, y_test)
    
#     # Create data loaders - 不打乱顺序
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#     val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # Using train for validation
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     # Show a sample batch
#     sample_x, sample_y = next(iter(train_loader))
#     print(f"Training batch - X shape: {sample_x.shape}, y shape: {sample_y.shape}")
#     print(f"Feature tensor format: [batch_size, channels, statistical_features] = [{sample_x.shape[0]}, {sample_x.shape[1]}, {sample_x.shape[2]}]")
#     print(f"Channels (sensors): {len(selected_sensors)}, Statistical features per channel: 9")
    
#     sample_test_x, sample_test_y = next(iter(test_loader))
#     print(f"Test batch - X shape: {sample_test_x.shape}, y shape: {sample_test_y.shape}")
    
#     return {
#         'train_loader': train_loader,
#         'val_loader': val_loader, 
#         'test_loader': test_loader,
#         'scalers': scalers,
#         'selected_sensors': selected_sensors,
#         'train_df': train_df,
#         'test_df': test_df,
#         'feature_names': ['mean', 'std', 'min', 'max', 'range', 'median', 'q25', 'q75', 'slope'],
#         'processing_config': {
#             'window_size': window_size,
#             'stride': stride,
#             'setting_normalization': setting_normalization,
#             'n_clusters_or_bins': n_clusters_or_bins,
#             'variance_threshold': variance_threshold,
#             'engine_level_normalization': engine_level_normalization,
#             'statistical_features': True,
#             'output_format': '[batch_size, channels, statistical_features]'
#         }
#     }

# # === Basic Python ===
# import os

# # === Data Handling ===
# import pandas as pd
# import numpy as np

# # === Plotting (可选，如果你做可视化分析) ===
# import matplotlib.pyplot as plt
# import seaborn as sns

# # === Machine Learning Utilities ===
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.cluster import KMeans

# # === Import additional libraries ===
# import numpy as np, torch
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import SelectKBest, f_regression
# from scipy.stats import pearsonr
# from scipy import stats
# import warnings
# warnings.filterwarnings('ignore')


# # === Comprehensive CMAPSS Data Processing Pipeline ===
# def process_cmapss_data(base_path="/content/turbofan_data", 
#                        fd_datasets=['FD001', 'FD002', 'FD003', 'FD004'],
#                        window_size=32, 
#                        stride=16,
#                        batch_size=64,
#                        setting_normalization='kmeans',  # 'kmeans' or 'binning'
#                        n_clusters_or_bins=5,
#                        variance_threshold=1e-6,
#                        selected_sensors=None,
#                        engine_level_normalization=True):
#     """
#     Comprehensive CMAPSS data processing pipeline with improved preprocessing
    
#     Args:
#         base_path: Path to CMAPSS data files
#         fd_datasets: List of FD datasets to process
#         window_size: Size of sliding windows
#         stride: Stride for sliding windows
#         batch_size: Batch size for data loaders
#         setting_normalization: Method for operational setting regularization ('kmeans' or 'binning')
#         n_clusters_or_bins: Number of clusters/bins for setting regularization
#         variance_threshold: Threshold for removing low-variance sensors
#         selected_sensors: Specific sensors to use (if None, auto-select based on variance)
#         engine_level_normalization: Whether to normalize at engine level vs global
#     """
    
#     # === Integrated CMAPSS data loading function ===
#     def load_cmapss_internal(base_path, dataset=None):
#         # Define column names for the CMAPSS data files (3 operational settings + 21 sensors)
#         cols = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor_{i}' for i in range(1, 22)]
        
#         # If a specific dataset is requested, process only that one
#         if dataset is not None:
#             # Extract dataset number from name (e.g., 'FD001' -> 1)
#             dataset_num = int(dataset[-1])
            
#             # Load training data for specified dataset
#             train = pd.read_csv(f"{base_path}/train_{dataset}.txt", sep='\s+', header=None, names=cols)

#             # Load test data and the corresponding RUL (remaining useful life) ground truth values
#             test = pd.read_csv(f"{base_path}/test_{dataset}.txt", sep='\s+', header=None, names=cols)
#             rul = pd.read_csv(f"{base_path}/RUL_{dataset}.txt", sep='\s+', header=None, names=['true_RUL'])

#             # Calculate the RUL for training data by subtracting each cycle from its engine's max cycle
#             max_cycle = train.groupby('engine_id')['cycle'].transform('max')  # Each engine's max cycle
#             train['RUL'] = max_cycle - train['cycle']  # RUL = max_cycle - current_cycle
#             train['dataset'] = dataset_num  # Label the subset ID

#             # Compute max cycle per engine in test set to determine current age
#             test['max_cycle'] = test.groupby('engine_id')['cycle'].transform('max')

#             # Assign RUL to each time step of each test engine based on true_RUL at final cycle
#             test['RUL'] = test.apply(
#                 lambda row: rul.iloc[int(row['engine_id']) - 1, 0] + (row['max_cycle'] - row['cycle']),
#                 axis=1
#             )

#             # Drop the temporary max_cycle column
#             test.drop('max_cycle', axis=1, inplace=True)
#             test['dataset'] = dataset_num  # Add subset label

#             # Clip RUL values only for training data: remove bottom 5% and top 5% of RUL values
#             train_rul_quantiles = train['RUL'].quantile([0.05, 0.95])
#             train = train[(train['RUL'] >= train_rul_quantiles[0.05]) & (train['RUL'] <= train_rul_quantiles[0.95])]
            
#             # Keep test data unclipped for real evaluation
            
#             return train, test
        
#         # Original behavior: load all datasets if no specific dataset is requested
#         # Initialize empty DataFrames to hold combined training and test data from all four subdatasets (FD001–FD004)
#         train_all, test_all = pd.DataFrame(), pd.DataFrame()

#         # Iterate through all four CMAPSS subsets (FD001 to FD004)
#         for i in range(1, 5):
#             # Load training data for current subset
#             train = pd.read_csv(f"{base_path}/train_FD00{i}.txt", sep='\s+', header=None, names=cols)

#             # Load test data and the corresponding RUL (remaining useful life) ground truth values
#             test = pd.read_csv(f"{base_path}/test_FD00{i}.txt", sep='\s+', header=None, names=cols)
#             rul = pd.read_csv(f"{base_path}/RUL_FD00{i}.txt", sep='\s+', header=None, names=['true_RUL'])

#             # Calculate the RUL for training data by subtracting each cycle from its engine's max cycle
#             max_cycle = train.groupby('engine_id')['cycle'].transform('max')  # Each engine's max cycle
#             train['RUL'] = max_cycle - train['cycle']  # RUL = max_cycle - current_cycle
#             train['dataset'] = i  # Label the subset ID (1 to 4)
#             train_all = pd.concat([train_all, train], ignore_index=True)  # Append to the global training set

#             # Compute max cycle per engine in test set to determine current age
#             test['max_cycle'] = test.groupby('engine_id')['cycle'].transform('max')

#             # To prevent duplicate engine IDs across different subsets, shift test engine IDs by offset
#             offset = test_all['engine_id'].max() + 1 if not test_all.empty else 0
#             test['engine_id'] += offset  # Shift engine_id to ensure uniqueness

#             # Assign RUL to each time step of each test engine based on true_RUL at final cycle
#             test['RUL'] = test.apply(
#                 lambda row: rul.iloc[int(row['engine_id']) - 1 - offset, 0] + (row['max_cycle'] - row['cycle']),
#                 axis=1
#             )

#             # Drop the temporary max_cycle column
#             test.drop('max_cycle', axis=1, inplace=True)
#             test['dataset'] = i  # Add subset label
#             test_all = pd.concat([test_all, test], ignore_index=True)  # Append to the global test set

#         # Clip RUL values only for training data: remove bottom 5% and top 5% of RUL values
#         train_rul_quantiles = train_all['RUL'].quantile([0.05, 0.95])
#         train_all = train_all[(train_all['RUL'] >= train_rul_quantiles[0.05]) & (train_all['RUL'] <= train_rul_quantiles[0.95])]
        
#         # Keep test data unclipped for real evaluation

#         # Return concatenated training and test DataFrames covering FD001 to FD004
#         return train_all, test_all
    
#     # === 1. Load raw data from specified FD datasets ===
#     train_dfs, test_dfs = [], []
    
#     for fd_name in fd_datasets:
#         train_df, test_df = load_cmapss_internal(base_path, dataset=fd_name)
        
#         # Add dataset identifier for tracking
#         train_df['dataset'] = fd_name
#         test_df['dataset'] = fd_name
        
#         train_dfs.append(train_df)
#         test_dfs.append(test_df)
    
#     # Combine all datasets
#     if len(train_dfs) > 1:
#         train_df = pd.concat(train_dfs, ignore_index=True)
#         test_df = pd.concat(test_dfs, ignore_index=True)
#     else:
#         train_df = train_dfs[0]
#         test_df = test_dfs[0]
    
#     # === 2. Operational Setting Regularization ===
#     setting_cols = ['setting1', 'setting2', 'setting3']
    
#     if setting_normalization == 'kmeans':
#         # Use KMeans clustering to discretize operational settings
#         kmeans = KMeans(n_clusters=n_clusters_or_bins, random_state=42)
        
#         # Fit on training data
#         train_settings = train_df[setting_cols].values
#         setting_clusters = kmeans.fit_predict(train_settings)
        
#         # Apply to test data
#         test_settings = test_df[setting_cols].values
#         test_clusters = kmeans.predict(test_settings)
        
#         # Create one-hot encoded features
#         for i in range(n_clusters_or_bins):
#             train_df[f'setting_cluster_{i}'] = (setting_clusters == i).astype(int)
#             test_df[f'setting_cluster_{i}'] = (test_clusters == i).astype(int)
        
#     elif setting_normalization == 'binning':
#         # Use binning for each setting dimension
#         for setting_col in setting_cols:
#             # Calculate bins based on training data quantiles
#             _, bins = pd.qcut(train_df[setting_col], q=n_clusters_or_bins, retbins=True, duplicates='drop')
            
#             # Apply binning to both train and test
#             train_df[f'{setting_col}_bin'] = pd.cut(train_df[setting_col], bins=bins, labels=False, include_lowest=True)
#             test_df[f'{setting_col}_bin'] = pd.cut(test_df[setting_col], bins=bins, labels=False, include_lowest=True)
            
#             # Create one-hot encoding
#             for i in range(len(bins)-1):
#                 train_df[f'{setting_col}_bin_{i}'] = (train_df[f'{setting_col}_bin'] == i).astype(int)
#                 test_df[f'{setting_col}_bin_{i}'] = (test_df[f'{setting_col}_bin'] == i).astype(int)
    
#     # === 3. Invalid Sensor Removal ===
#     sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
#     valid_sensors = []
    
#     for sensor_col in sensor_cols:
#         # Calculate variance across all engines and cycles
#         sensor_variance = train_df[sensor_col].var()
        
#         if sensor_variance > variance_threshold:
#             valid_sensors.append(sensor_col)
    
#     # Update selected sensors if not provided
#     if selected_sensors is None:
#         # Convert to 1-based indexing for sensor numbers
#         selected_sensors = [int(s.split('_')[1]) for s in valid_sensors]
#     else:
#         # Filter selected sensors to only include valid ones
#         valid_sensor_numbers = [int(s.split('_')[1]) for s in valid_sensors]
#         selected_sensors = [s for s in selected_sensors if s in valid_sensor_numbers]
    
#     # === 4. Engine-level Normalization ===
#     sensor_cols_to_use = [f'sensor_{i}' for i in selected_sensors]
#     scalers = {}
    
#     if engine_level_normalization:
#         # Normalize each sensor for each engine separately
#         for sensor_col in sensor_cols_to_use:
#             scalers[sensor_col] = {}
            
#             # For each engine, fit a separate scaler
#             for engine_id in train_df['engine_id'].unique():
#                 engine_data = train_df[train_df['engine_id'] == engine_id][sensor_col].values.reshape(-1, 1)
                
#                 if len(engine_data) > 1 and engine_data.std() > 1e-8:
#                     scaler = StandardScaler()
#                     scaler.fit(engine_data)
#                     scalers[sensor_col][engine_id] = scaler
                    
#                     # Apply to training data
#                     train_df.loc[train_df['engine_id'] == engine_id, sensor_col] = scaler.transform(engine_data).flatten()
#                 else:
#                     # Handle constant or near-constant sequences
#                     scalers[sensor_col][engine_id] = None
            
#             # For test data, use the closest training engine's scaler or global fallback
#             global_scaler = StandardScaler()
#             global_scaler.fit(train_df[sensor_col].values.reshape(-1, 1))
            
#             for engine_id in test_df['engine_id'].unique():
#                 engine_data = test_df[test_df['engine_id'] == engine_id][sensor_col].values.reshape(-1, 1)
                
#                 # Try to find a corresponding training scaler or use global
#                 if engine_id in scalers[sensor_col] and scalers[sensor_col][engine_id] is not None:
#                     scaler = scalers[sensor_col][engine_id]
#                 else:
#                     scaler = global_scaler
                
#                 test_df.loc[test_df['engine_id'] == engine_id, sensor_col] = scaler.transform(engine_data).flatten()
        
#     else:
#         # Global normalization
#         for sensor_col in sensor_cols_to_use:
#             scaler = StandardScaler()
            
#             # Fit on training data
#             train_data = train_df[sensor_col].values.reshape(-1, 1)
#             train_df[sensor_col] = scaler.fit_transform(train_data).flatten()
            
#             # Apply to test data
#             test_data = test_df[sensor_col].values.reshape(-1, 1)
#             test_df[sensor_col] = scaler.transform(test_data).flatten()
            
#             scalers[sensor_col] = scaler
    
#     # === 5. Statistical Feature Extraction from Sliding Windows ===
#     def extract_statistical_features(window_data):
#         """
#         对滑窗内的每个通道在整个窗口T内做统计分析
#         输入: window_data - shape [T, F] (时间步，特征数)
#         输出: features - shape [F, 统计特征数] (每个传感器的统计特征向量)
#         """
#         T, F = window_data.shape
#         num_stats = 9  # 统计特征数量
#         features = np.zeros((F, num_stats))
        
#         for f in range(F):
#             channel_data = window_data[:, f]
            
#             # 基础统计特征
#             mean_val = np.mean(channel_data)                    # 均值 - 表示整体水平
#             std_val = np.std(channel_data)                      # 标准差 - 波动范围，反映状态稳定性
#             min_val = np.min(channel_data)                      # 最小值 - 捕捉尖峰或故障信号
#             max_val = np.max(channel_data)                      # 最大值 - 捕捉尖峰或故障信号
#             range_val = max_val - min_val                       # 变化幅度
#             median_val = np.median(channel_data)                # 中位数 - 抗异常值
#             q25 = np.percentile(channel_data, 25)               # 25分位数
#             q75 = np.percentile(channel_data, 75)               # 75分位数
            
#             # 线性趋势 - 表征退化方向
#             time_indices = np.arange(T)
#             if T > 1:
#                 slope, _, _, _, _ = stats.linregress(time_indices, channel_data)
#             else:
#                 slope = 0
            
#             # 存储为 [通道, 统计特征] 格式
#             features[f, :] = [mean_val, std_val, min_val, max_val, range_val, 
#                              median_val, q25, q75, slope]
        
#         return features
    
#     def create_windows_with_statistical_features(data, window_size, stride, sensor_indices):
#         """创建滑动窗口并提取统计特征，返回 [batch, 通道, 统计特征] 格式，保持时间顺序"""
#         X, y, window_indices = [], [], []
#         sensors = [f'sensor_{i}' for i in sensor_indices]
        
#         # 保持引擎ID的原始顺序，不打乱
#         for eid in sorted(data['engine_id'].unique()):
#             series = data[data['engine_id'] == eid].sort_values('cycle')
#             s_vals = series[sensors].values
#             ruls = series['RUL'].values
            
#             # Create sliding windows，保持时间顺序
#             for i in range(0, len(s_vals) - window_size + 1, stride):
#                 window_data = s_vals[i:i+window_size]
#                 window_rul = ruls[i + window_size - 1]  # RUL at the end of window
                
#                 # Clean data
#                 window_data = np.nan_to_num(window_data, nan=0.0, posinf=1e6, neginf=-1e6)
#                 window_rul = np.nan_to_num(window_rul, nan=0.0, posinf=1e6, neginf=-1e6)
                
#                 # 提取统计特征 - 从 [T, F] 到 [F, 统计特征数]
#                 statistical_features = extract_statistical_features(window_data)
                
#                 X.append(statistical_features)
#                 y.append(window_rul)
#                 window_indices.append((eid, i))
        
#         return np.array(X), np.array(y), window_indices
    
#     # Create windows with statistical features for training and test data
#     X_train, y_train, train_indices = create_windows_with_statistical_features(
#         train_df, window_size, stride, selected_sensors
#     )
    
#     X_test, y_test, test_indices = create_windows_with_statistical_features(
#         test_df, window_size, stride, selected_sensors
#     )
    
#     # === 6. Final Data Cleaning and Clipping ===
#     # Clip RUL values to reasonable ranges
#     y_train = np.clip(y_train, 0.0, 1e6)
#     y_test = np.clip(y_test, 0.0, 1e6)
    
#     # Clamp feature data to prevent CUDA errors
#     X_train = np.clip(X_train, -1e6, 1e6)
#     X_test = np.clip(X_test, -1e6, 1e6)
    
#     # === 7. Create PyTorch Datasets and DataLoaders ===
#     class RULDataset(Dataset):
#         def __init__(self, X, y):
#             self.X = torch.tensor(X, dtype=torch.float32)
#             self.y = torch.tensor(y, dtype=torch.float32)
            
#         def __len__(self): 
#             return len(self.X)
        
#         def __getitem__(self, idx):
#             # Return tuple instead of dictionary to avoid warnings
#             return self.X[idx], self.y[idx]
    
#     # Create datasets
#     train_dataset = RULDataset(X_train, y_train)
#     test_dataset = RULDataset(X_test, y_test)
    
#     # Create data loaders - 不打乱顺序
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#     val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # Using train for validation
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     # Show a sample batch
#     sample_x, sample_y = next(iter(train_loader))
#     print(f"Training batch - X shape: {sample_x.shape}, y shape: {sample_y.shape}")
#     print(f"Feature tensor format: [batch_size, channels, statistical_features] = [{sample_x.shape[0]}, {sample_x.shape[1]}, {sample_x.shape[2]}]")
#     print(f"Channels (sensors): {len(selected_sensors)}, Statistical features per channel: 9")
    
#     sample_test_x, sample_test_y = next(iter(test_loader))
#     print(f"Test batch - X shape: {sample_test_x.shape}, y shape: {sample_test_y.shape}")
    
#     return {
#         'train_loader': train_loader,
#         'val_loader': val_loader, 
#         'test_loader': test_loader,
#         'scalers': scalers,
#         'selected_sensors': selected_sensors,
#         'train_df': train_df,
#         'test_df': test_df,
#         'feature_names': ['mean', 'std', 'min', 'max', 'range', 'median', 'q25', 'q75', 'slope'],
#         'processing_config': {
#             'window_size': window_size,
#             'stride': stride,
#             'setting_normalization': setting_normalization,
#             'n_clusters_or_bins': n_clusters_or_bins,
#             'variance_threshold': variance_threshold,
#             'engine_level_normalization': engine_level_normalization,
#             'statistical_features': True,
#             'output_format': '[batch_size, channels, statistical_features]'
#         }
#     }
