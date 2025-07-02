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

    if not os.path.exists(os.path.join(extract_path, "train_FD001.txt")):
        print("Downloading C-MAPSS dataset...")
        response = requests.get(GITHUB_URL)
        with open(zip_path, 'wb') as f:
            f.write(response.content)

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        os.remove(zip_path)
        print("Dataset downloaded and extracted to:", extract_path)
    else:
        print("C-MAPSS data already exists, skipping download.")

    # 2. Load datasets
    print("\nLoading datasets...")
    columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
              [f'sensor_{i}' for i in range(1, 22)]

    datasets = {}

    for fd in ['001', '002', '003', '004']:
        prefix = f'FD{fd}'
        train_file = os.path.join(extract_path, f"train_{prefix}.txt")
        test_file = os.path.join(extract_path, f"test_{prefix}.txt")
        rul_file = os.path.join(extract_path, f"RUL_{prefix}.txt")

        if not (os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(rul_file)):
            print(f"Missing files for {prefix}")
            continue

        train_df = pd.read_csv(train_file, sep=' ', header=None)
        train_df = train_df.dropna(axis=1)
        train_df.columns = columns
        max_cycles = train_df.groupby("engine_id")["cycle"].max()
        train_df["RUL"] = train_df.apply(lambda row: max_cycles[row["engine_id"]] - row["cycle"], axis=1)
        train_rul = train_df.groupby("engine_id")["RUL"].last().reset_index()

        test_df = pd.read_csv(test_file, sep=' ', header=None)
        test_df = test_df.dropna(axis=1)
        test_df.columns = columns

        test_rul = pd.read_csv(rul_file, sep=' ', header=None, names=["RUL"])
        test_rul["engine_id"] = range(1, len(test_rul) + 1)

        datasets[prefix] = {
            "train_data": train_df,
            "test_data": test_df,
            "train_rul": train_rul,
            "test_rul": test_rul
        }

        print(f"{prefix} loaded: Train={train_df.shape}, Test={test_df.shape}")

    return datasets
