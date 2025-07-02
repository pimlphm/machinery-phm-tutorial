import os
import zipfile
import pandas as pd
import requests

def load_turbofan_data(extract_path="turbofan_data"):
    """
    Download and process C-MAPSS turbofan engine dataset
    
    Args:
        extract_path: Path to store the data
    
    Returns:
        dict: Dictionary containing four datasets, each with train_data, test_data, train_rul, test_rul
    """
    # GitHub dataset link
    url = "https://raw.githubusercontent.com/pimlphm/machinery-phm-tutorial/main/data_bases/prognostics/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
    
    print("Downloading C-MAPSS dataset...")
    
    # Create directory
    os.makedirs(extract_path, exist_ok=True)
    
    # Download and extract
    zip_path = os.path.join(extract_path, "dataset.zip")
    response = requests.get(url)
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    os.remove(zip_path)
    
    print("Dataset downloaded successfully, processing...")
    
    # Define column names
    columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    datasets = {}
    
    # Process four datasets
    for fd_num in ['001', '002', '003', '004']:
        dataset_name = f"FD{fd_num}"
        
        # Find files
        train_file = test_file = rul_file = None
        for root, _, files in os.walk(extract_path):
            for file in files:
                if f"train_FD{fd_num}.txt" in file:
                    train_file = os.path.join(root, file)
                elif f"test_FD{fd_num}.txt" in file:
                    test_file = os.path.join(root, file)
                elif f"RUL_FD{fd_num}.txt" in file:
                    rul_file = os.path.join(root, file)
        
        # Read training data
        train_data = pd.read_csv(train_file, sep=' ', header=None, names=columns)
        train_data = train_data.dropna(axis=1, how='all')
        
        # Calculate RUL for training data
        max_cycles = train_data.groupby('engine_id')['cycle'].max()
        train_data['RUL'] = train_data.apply(lambda x: max_cycles[x['engine_id']] - x['cycle'], axis=1)
        
        # Read test data
        test_data = pd.read_csv(test_file, sep=' ', header=None, names=columns)
        test_data = test_data.dropna(axis=1, how='all')
        
        # Read test RUL
        test_rul = pd.read_csv(rul_file, sep=' ', header=None, names=['RUL'])
        test_rul['engine_id'] = range(1, len(test_rul) + 1)
        
        # Get training RUL (last cycle RUL for each engine)
        train_rul = train_data.groupby('engine_id')['RUL'].last().reset_index()
        
        datasets[dataset_name] = {
            'train_data': train_data,
            'test_data': test_data,
            'train_rul': train_rul,
            'test_rul': test_rul
        }
        
        print(f"{dataset_name} processed - Training set: {train_data.shape}, Test set: {test_data.shape}")
    
    print(f"All datasets processed successfully, saved in: {extract_path}")
    return datasets
if __name__ == "__main__":
    datasets = load_turbofan_data('/content/turbofan_data')
