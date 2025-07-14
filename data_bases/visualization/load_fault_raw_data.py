import numpy as np

def load_fault_data(file_path='processed_fault_data.npz', num_samples=300, verbose=True):
    """
    Load fault data from .npz file
    
    Args:
        file_path (str): Path to the .npz file
        num_samples (int): Number of samples to load for each fault type
        verbose (bool): Whether to print information about the data
    
    Returns:
        dict: Dictionary containing the loaded data with keys:
              'normal', 'misalignment', 'unbalance', 'looseness', 'fs', 'rpm'
    """
    # Open .npz file without loading data into memory
    if verbose:
        with np.load(file_path, mmap_mode='r') as data:
            print("📁 Keys and shapes:")
            for key in data.files:
                print(f"- {key}: shape = {data[key].shape}, dtype = {data[key].dtype}")

    with np.load(file_path) as data:
        normal_data     = data['normal'][:num_samples].copy()
        misalign_data   = data['misalignment'][:num_samples].copy()
        unbalance_data  = data['unbalance'][:num_samples].copy()
        looseness_data  = data['looseness'][:num_samples].copy()
        fs = data['fs'].item()
        rpm = data['rpm'].item()

    # # Now everything else in .npz is released from memory
    # if verbose:
    #     print("✅ Shapes after loading subset:")
    #     print(f"- Normal:       {normal_data.shape}")
    #     print(f"- Misalignment: {misalign_data.shape}")
    #     print(f"- Unbalance:    {unbalance_data.shape}")
    #     print(f"- Looseness:    {looseness_data.shape}")
    #     print(f"- fs: {fs} Hz, rpm: {rpm} rpm")

    return normal_data,misalign_data, unbalance_data,  looseness_data, fs,rpm

