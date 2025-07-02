import os
import zipfile
import pandas as pd
import requests
from urllib.parse import urljoin

# === Configuration ===
GITHUB_RAW_URL = "https://raw.githubusercontent.com/pimlphm/machinery-phm-tutorial/main/data_bases/prognostics/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
DEFAULT_EXTRACT_PATH = "turbofan_data"

def download_and_extract_turbofan_data(extract_path=DEFAULT_EXTRACT_PATH):
    """Download and extract C-MAPSS turbofan dataset from GitHub repository."""
    print("Downloading C-MAPSS Turbofan dataset...")
    
    # Create extract directory
    os.makedirs(extract_path, exist_ok=True)
    
    # Download zip file
    zip_path = os.path.join(extract_path, "turbofan_dataset.zip")
    try:
        response = requests.get(GITHUB_RAW_URL)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print("Dataset downloaded successfully")
        
    except Exception as e:
        print(f"Download failed: {e}")
        return None
    
    # Extract zip file
    print("Extracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Dataset extracted successfully")
        
        # Remove zip file after extraction
        os.remove(zip_path)
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None
    
    return extract_path

def load_cmapss_datasets(extract_path=DEFAULT_EXTRACT_PATH):
    """Load all C-MAPSS dataset variants and save processed data."""
    print("\nLoading C-MAPSS datasets...")
    
    # Define column names
    columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
              [f'sensor_{i}' for i in range(1, 22)]
    
    datasets = {}
    
    # Process each dataset variant (FD001, FD002, FD003, FD004)
    for fd_num in ['001', '002', '003', '004']:
        dataset_name = f"FD{fd_num}"
        print(f"\nProcessing {dataset_name}...")
        
        # Find files in extracted directory
        train_file = None
        test_file = None
        rul_file = None
        
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if f"train_FD{fd_num}.txt" in file:
                    train_file = os.path.join(root, file)
                elif f"test_FD{fd_num}.txt" in file:
                    test_file = os.path.join(root, file)
                elif f"RUL_FD{fd_num}.txt" in file:
                    rul_file = os.path.join(root, file)
        
        if not all([train_file, test_file, rul_file]):
            print(f"Missing files for {dataset_name}")
            continue
        
        # Load training data
        train_data = pd.read_csv(train_file, sep=' ', header=None, names=columns)
        train_data = train_data.dropna(axis=1, how='all')  # Remove empty columns
        
        # Calculate RUL for training data
        train_rul = train_data.groupby('engine_id')['cycle'].max().reset_index()
        train_rul.columns = ['engine_id', 'max_cycle']
        train_data = train_data.merge(train_rul, on='engine_id')
        train_data['RUL'] = train_data['max_cycle'] - train_data['cycle']
        train_data = train_data.drop('max_cycle', axis=1)
        
        # Load test data
        test_data = pd.read_csv(test_file, sep=' ', header=None, names=columns)
        test_data = test_data.dropna(axis=1, how='all')
        
        # Load test RUL
        test_rul = pd.read_csv(rul_file, sep=' ', header=None, names=['RUL'])
        test_rul['engine_id'] = range(1, len(test_rul) + 1)
        
        # Save processed data
        output_dir = os.path.join(extract_path, f"processed_{dataset_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        train_data.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
        train_data[['engine_id', 'RUL']].groupby('engine_id').last().reset_index().to_csv(
            os.path.join(output_dir, "train_rul.csv"), index=False)
        test_data.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)
        test_rul.to_csv(os.path.join(output_dir, "test_rul.csv"), index=False)
        
        datasets[dataset_name] = {
            'train_data': train_data,
            'train_rul': train_data[['engine_id', 'RUL']].groupby('engine_id').last().reset_index(),
            'test_data': test_data,
            'test_rul': test_rul
        }
        
        print(f"{dataset_name} processed - Train: {train_data.shape}, Test: {test_data.shape}")
    
    return datasets

def print_dataset_info():
    """Print information about the C-MAPSS dataset."""
    print("\n" + "="*60)
    print("C-MAPSS TURBOFAN ENGINE DATASET")
    print("="*60)
    
    print("\nDESCRIPTION:")
    print("The C-MAPSS dataset contains run-to-failure simulations of")
    print("turbofan engines with sensor measurements and operating conditions.")
    print("It's widely used for prognostics and health management research.")
    
    print("\nDATASET VARIANTS:")
    variants = [
        ("FD001", "Single fault mode, single operating condition"),
        ("FD002", "Single fault mode, six operating conditions"), 
        ("FD003", "Two fault modes, single operating condition"),
        ("FD004", "Two fault modes, six operating conditions")
    ]
    
    for variant, description in variants:
        print(f"• {variant}: {description}")
    
    print("\nDATA STRUCTURE:")
    print("• engine_id: Unique engine identifier")
    print("• cycle: Operational cycle number")
    print("• setting1-3: Operational settings")
    print("• sensor_1-21: Sensor measurements")
    print("• RUL: Remaining Useful Life (target variable)")
    
    print("\nAPPLICATIONS:")
    print("• Predictive maintenance")
    print("• Remaining useful life prediction")
    print("• Fault detection and diagnosis")
    print("• Health monitoring systems")
    
    print("="*60)

def main(extract_path=DEFAULT_EXTRACT_PATH):
    """Main function to download, extract, and process C-MAPSS dataset."""
    print_dataset_info()
    
    # Download and extract data
    if download_and_extract_turbofan_data(extract_path):
        # Load and process datasets
        datasets = load_cmapss_datasets(extract_path)
        
        if datasets:
            print(f"\nAll datasets processed and saved to: {extract_path}")
            print("\nProcessed files structure:")
            for fd_name in datasets.keys():
                print(f"processed_{fd_name}/")
                print("  ├── train_data.csv")
                print("  ├── train_rul.csv") 
                print("  ├── test_data.csv")
                print("  └── test_rul.csv")
            
            return datasets
        else:
            print("Failed to process datasets")
            return None
    else:
        print("Failed to download dataset")
        return None

# === 使用方法 ===
# 1. 下载此文件:
# !wget https://raw.githubusercontent.com/pimlphm/machinery-phm-tutorial/main/data_bases/visualization/turbofan_data_loader.py -O turbofan_data_loader.py

# 2. 导入和使用:
# from turbofan_data_loader import download_and_extract_turbofan_data, load_cmapss_datasets, main

# === Execution ===
if __name__ == "__main__":
    datasets = main()
