import os
import zipfile
import pandas as pd
import requests

def load_turbofan_data(extract_path="turbofan_data"):
    """
    Download, extract, and load the full C-MAPSS Turbofan dataset.

    Args:
        extract_path (str): Local directory to store and load the data.

    Returns:
        dict: Dictionary with keys ['FD001', 'FD002', 'FD003', 'FD004'],
              each containing a dict with keys: train_data, test_data, train_rul, test_rul.
    """
    # 1. Download and extract
    GITHUB_URL = "https://raw.githubusercontent.com/pimlphm/machinery-phm-tutorial/main/data_bases/prognostics/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
    os.makedirs(extract_path, exist_ok=True)
    zip_path = os.path.join(extract_path, "CMAPSS.zip")

    # Check if data files already exist
    existing_files = []
    for root, _, files in os.walk(extract_path):
        existing_files.extend(files)
    
    has_train_files = any("train_FD001.txt" in f for f in existing_files)
    
    if not has_train_files:
        print("Downloading C-MAPSS dataset...")
        response = requests.get(GITHUB_URL)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            f.write(response.content)

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Extract the inner CMAPSSData.zip if it exists
        inner_zip_path = None
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file.lower() == "cmapssdata.zip":
                    inner_zip_path = os.path.join(root, file)
                    break
            if inner_zip_path:
                break
                
        if inner_zip_path:
            print("Extracting inner CMAPSSData.zip...")
            with zipfile.ZipFile(inner_zip_path, 'r') as inner_zip_ref:
                inner_zip_ref.extractall(extract_path)
            # Remove the inner zip file
            os.remove(inner_zip_path)
        
        # Clean up outer zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print(f"Dataset downloaded and extracted to: {extract_path}")
    else:
        print("C-MAPSS data already exists, skipping download.")

    # 2. Load datasets
    print("\nLoading datasets...")
    columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
              [f'sensor_{i}' for i in range(1, 22)]

    datasets = {}

    for fd in ['001', '002', '003', '004']:
        prefix = f'FD{fd}'

        # Search for files in all subdirectories
        train_file = test_file = rul_file = None
        for root, _, files in os.walk(extract_path):
            for fname in files:
                full_path = os.path.join(root, fname)
                if fname == f"train_{prefix}.txt" or fname.lower() == f"train_{prefix}.txt":
                    train_file = full_path
                elif fname == f"test_{prefix}.txt" or fname.lower() == f"test_{prefix}.txt":
                    test_file = full_path
                elif fname == f"RUL_{prefix}.txt" or fname.lower() == f"rul_{prefix}.txt":
                    rul_file = full_path

        # Debug: print found files
        print(f"Looking for {prefix} files:")
        print(f"  Train file: {train_file}")
        print(f"  Test file: {test_file}")
        print(f"  RUL file: {rul_file}")

        if not (train_file and test_file and rul_file):
            print(f"Warning: Missing files for {prefix}")
            # List all files in extract_path for debugging
            print(f"Available files in {extract_path}:")
            for root, _, files in os.walk(extract_path):
                for f in files:
                    if prefix.lower() in f.lower():
                        print(f"  {os.path.join(root, f)}")
            continue

        try:
            # Read train data
            train_df = pd.read_csv(train_file, sep=' ', header=None)
            train_df = train_df.dropna(axis=1, how='all')  # Remove completely empty columns
            
            # Ensure we have the right number of columns
            if train_df.shape[1] != len(columns):
                print(f"Warning: Expected {len(columns)} columns, got {train_df.shape[1]} for {prefix} train data")
                # Adjust columns to match actual data
                actual_columns = columns[:train_df.shape[1]]
                train_df.columns = actual_columns
            else:
                train_df.columns = columns
            
            # Calculate RUL for training data
            max_cycles = train_df.groupby("engine_id")["cycle"].max()
            train_df["RUL"] = train_df.apply(lambda row: max_cycles[row["engine_id"]] - row["cycle"], axis=1)
            train_rul = train_df.groupby("engine_id")["RUL"].last().reset_index()

            # Read test data
            test_df = pd.read_csv(test_file, sep=' ', header=None)
            test_df = test_df.dropna(axis=1, how='all')  # Remove completely empty columns
            
            # Ensure we have the right number of columns
            if test_df.shape[1] != len(columns):
                print(f"Warning: Expected {len(columns)} columns, got {test_df.shape[1]} for {prefix} test data")
                # Adjust columns to match actual data
                actual_columns = columns[:test_df.shape[1]]
                test_df.columns = actual_columns
            else:
                test_df.columns = columns

            # Read RUL data
            test_rul = pd.read_csv(rul_file, sep=' ', header=None, names=["RUL"])
            test_rul = test_rul.dropna(axis=1, how='all')  # Remove completely empty columns
            if test_rul.shape[1] == 1:
                test_rul.columns = ["RUL"]
            test_rul["engine_id"] = range(1, len(test_rul) + 1)

            datasets[prefix] = {
                "train_data": train_df,
                "test_data": test_df,
                "train_rul": train_rul,
                "test_rul": test_rul
            }

            print(f"{prefix} loaded successfully: Train={train_df.shape}, Test={test_df.shape}")

        except Exception as e:
            print(f"Error loading {prefix}: {str(e)}")
            continue

    if not datasets:
        print("No datasets were loaded successfully. Please check the file structure.")
        # List all files for debugging
        print("All files in extract_path:")
        for root, _, files in os.walk(extract_path):
            for f in files:
                print(f"  {os.path.join(root, f)}")
    
    # Extract data as numpy arrays
    train_data = {}
    test_data = {}
    train_rul = {}
    test_rul = {}

    for fd_key in datasets.keys():
        train_data[fd_key] = datasets[fd_key]['train_data'].values
        test_data[fd_key] = datasets[fd_key]['test_data'].values
        train_rul[fd_key] = datasets[fd_key]['train_rul'].values
        test_rul[fd_key] = datasets[fd_key]['test_rul'].values

    print("\nData extraction completed:")
    for fd_key in datasets.keys():
        print(f"{fd_key}: Train data shape: {train_data[fd_key].shape}, Test data shape: {test_data[fd_key].shape}")
    
    # Print detailed dimension information
    print("\n" + "="*80)
    print("DETAILED DIMENSION INFORMATION")
    print("="*80)
    
    # Setting参数的物理含义
    setting_descriptions = {
        'setting1': 'Altitude (operational altitude setting)',
        'setting2': 'Mach number (speed setting)',
        'setting3': 'Throttle resolver angle (power setting)'
    }
    
    # Sensor传感器的物理含义
    sensor_descriptions = {
        'sensor_1': 'Total temperature at fan inlet (T2)',
        'sensor_2': 'Total temperature at LPC outlet (T24)',
        'sensor_3': 'Total temperature at HPC outlet (T30)',
        'sensor_4': 'Total temperature at LPT outlet (T50)',
        'sensor_5': 'Pressure at fan inlet (P2)',
        'sensor_6': 'Total pressure in bypass-duct (P15)',
        'sensor_7': 'Total pressure at HPC outlet (P30)',
        'sensor_8': 'Physical fan speed (Nf)',
        'sensor_9': 'Physical core speed (Nc)',
        'sensor_10': 'Engine pressure ratio (P30/P2)',
        'sensor_11': 'Static pressure at HPC outlet (Ps30)',
        'sensor_12': 'Ratio of fuel flow to Ps30',
        'sensor_13': 'Corrected fan speed',
        'sensor_14': 'Corrected core speed',
        'sensor_15': 'Bypass Ratio',
        'sensor_16': 'Burner fuel-air ratio',
        'sensor_17': 'Bleed Enthalpy',
        'sensor_18': 'Required fan speed',
        'sensor_19': 'Required core speed',
        'sensor_20': 'High pressure turbine efficiency',
        'sensor_21': 'Low pressure turbine efficiency'
    }
    
    for fd_key in datasets.keys():
        print(f"\n{fd_key} Dataset:")
        train_df = datasets[fd_key]['train_data']
        test_df = datasets[fd_key]['test_data']
        
        print(f"\nTrain Data ({train_df.shape}):")
        print(f"  Number of dimensions: {train_df.shape[1]}")
        print(f"  Dimension details:")
        for i, col in enumerate(train_df.columns):
            print(f"    Dimension {i+1}: {col}")
            if col == 'engine_id':
                print(f"      - Description: Engine identifier")
                print(f"      - Value range: {train_df[col].min()} to {train_df[col].max()}")
                print(f"      - Number of unique engines: {train_df[col].nunique()}")
            elif col == 'cycle':
                print(f"      - Description: Time cycle (operational cycle)")
                print(f"      - Value range: {train_df[col].min()} to {train_df[col].max()}")
                print(f"      - Average cycles per engine: {train_df[col].max() / train_df['engine_id'].nunique():.1f}")
            elif col.startswith('setting'):
                desc = setting_descriptions.get(col, 'Operational setting parameter')
                print(f"      - Description: {desc}")
                print(f"      - Value range: {train_df[col].min():.4f} to {train_df[col].max():.4f}")
                print(f"      - Mean: {train_df[col].mean():.4f}, Std: {train_df[col].std():.4f}")
                print(f"      - Number of unique values: {train_df[col].nunique()}")
            elif col.startswith('sensor_'):
                desc = sensor_descriptions.get(col, 'Sensor measurement')
                print(f"      - Description: {desc}")
            elif col == 'RUL':
                print(f"      - Description: Remaining Useful Life (target variable)")
                print(f"      - Value range: {train_df[col].min()} to {train_df[col].max()}")
                print(f"      - Mean: {train_df[col].mean():.2f}, Std: {train_df[col].std():.2f}")
        
        print(f"\nTest Data ({test_df.shape}):")
        print(f"  Number of dimensions: {test_df.shape[1]}")
        print(f"  Dimension details:")
        for i, col in enumerate(test_df.columns):
            print(f"    Dimension {i+1}: {col}")
            if col == 'engine_id':
                print(f"      - Description: Engine identifier")
                print(f"      - Value range: {test_df[col].min()} to {test_df[col].max()}")
                print(f"      - Number of unique engines: {test_df[col].nunique()}")
            elif col == 'cycle':
                print(f"      - Description: Time cycle (operational cycle)")
                print(f"      - Value range: {test_df[col].min()} to {test_df[col].max()}")
                print(f"      - Average cycles per engine: {test_df[col].max() / test_df['engine_id'].nunique():.1f}")
            elif col.startswith('setting'):
                desc = setting_descriptions.get(col, 'Operational setting parameter')
                print(f"      - Description: {desc}")
                print(f"      - Value range: {test_df[col].min():.4f} to {test_df[col].max():.4f}")
                print(f"      - Mean: {test_df[col].mean():.4f}, Std: {test_df[col].std():.4f}")
                print(f"      - Number of unique values: {test_df[col].nunique()}")
            elif col.startswith('sensor_'):
                desc = sensor_descriptions.get(col, 'Sensor measurement')
                print(f"      - Description: {desc}")
        
        print(f"\n  Note: Test data has {test_df.shape[1]} dimensions (missing RUL column)")
        print(f"        Train data has {train_df.shape[1]} dimensions (includes RUL column)")
    
    print("\n" + "="*80)
    # print("SETTING PARAMETERS DETAILED INFORMATION:")
    # print("="*80)
    # for setting, desc in setting_descriptions.items():
    #     print(f"\n{setting.upper()}:")
    #     print(f"  Physical meaning: {desc}")
        
    #     # 统计各数据集中该setting的信息
    #     for fd_key in datasets.keys():
    #         train_df = datasets[fd_key]['train_data']
    #         if setting in train_df.columns:
    #             unique_vals = sorted(train_df[setting].unique())
    #             print(f"  {fd_key} - Unique values: {unique_vals}")
    #             print(f"  {fd_key} - Value distribution:")
    #             value_counts = train_df[setting].value_counts().sort_index()
    #             for val, count in value_counts.items():
    #                 print(f"    {val}: {count} samples ({count/len(train_df)*100:.1f}%)")
    
    # print("\n" + "="*80)
    # print("SENSOR MEASUREMENTS DETAILED INFORMATION:")
    # print("="*80)
    # for sensor, desc in sensor_descriptions.items():
    #     print(f"\n{sensor.upper()}:")
    #     print(f"  Physical meaning: {desc}")
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("Train data dimensions (27):")
    print("  1. engine_id - Engine identifier")
    print("  2. cycle - Time cycle")
    print("  3-5. setting1, setting2, setting3 - Operational settings")
    print("  6-26. sensor_1 to sensor_21 - 21 sensor measurements")
    print("  27. RUL - Remaining Useful Life (target variable)")
    print("\nTest data dimensions (26):")
    print("  1. engine_id - Engine identifier")
    print("  2. cycle - Time cycle")
    print("  3-5. setting1, setting2, setting3 - Operational settings")
    print("  6-26. sensor_1 to sensor_21 - 21 sensor measurements")
    print("  Note: RUL is provided separately in RUL files for test data")
    print("="*80)
    
    return train_data,test_data,train_rul,test_rul
