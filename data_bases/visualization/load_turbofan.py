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
        
        # Clean up zip file
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
    
    return datasets
