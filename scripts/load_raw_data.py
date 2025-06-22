# === Path Setup (Colab + Google Drive) ===
import os
import zipfile
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output

# === User Interactive Interface for Dataset Selection ===
def create_dataset_selector():
    """Create dataset selection interface"""
    
    # Scan available datasets
    data_bases_path = "machinery-phm-tutorial/data_bases"
    
    # Get prognostics and diagnostics folders
    category_options = []
    if os.path.exists(os.path.join(data_bases_path, "prognostics")):
        category_options.append("prognostics")
    if os.path.exists(os.path.join(data_bases_path, "diagnostics")):
        category_options.append("diagnostics")
    
    # Create interface components
    category_dropdown = widgets.Dropdown(
        options=category_options,
        value=category_options[0] if category_options else None,
        description='Data Type:',
        style={'description_width': 'initial'}
    )
    
    dataset_dropdown = widgets.Dropdown(
        options=[],
        description='Dataset:',
        style={'description_width': 'initial'}
    )
    
    extract_path_text = widgets.Text(
        value='extracted_data',
        description='Extract Path:',
        style={'description_width': 'initial'}
    )
    
    confirm_button = widgets.Button(
        description='Confirm Selection',
        button_style='success',
        icon='check'
    )
    
    output_area = widgets.Output()
    
    # Function to update dataset options
    def update_datasets(change):
        category = change['new']
        category_path = os.path.join(data_bases_path, category)
        
        datasets = []
        if os.path.exists(category_path):
            for file in os.listdir(category_path):
                if file.endswith('.zip'):
                    datasets.append(file)
        
        dataset_dropdown.options = datasets
        if datasets:
            dataset_dropdown.value = datasets[0]
    
    # Bind events
    category_dropdown.observe(update_datasets, names='value')
    
    # Initialize dataset options
    if category_options:
        update_datasets({'new': category_options[0]})
    
    # Confirm button click event
    def on_confirm_click(b):
        with output_area:
            clear_output()
            
            selected_category = category_dropdown.value
            selected_dataset = dataset_dropdown.value
            extract_path = extract_path_text.value
            
            if not selected_category or not selected_dataset:
                print("❌ Please select valid data type and dataset")
                return
            
            print(f"✅ Selected:")
            print(f"   📂 Data Type: {selected_category}")
            print(f"   📄 Dataset: {selected_dataset}")
            print(f"   📁 Extract Path: {extract_path}")
            
            # Execute data extraction
            extract_selected_dataset(selected_category, selected_dataset, extract_path)
    
    confirm_button.on_click(on_confirm_click)
    
    # Display interface
    print("🔧 Please select the dataset to use:")
    display(widgets.VBox([
        category_dropdown,
        dataset_dropdown,
        extract_path_text,
        confirm_button,
        output_area
    ]))
    
    return category_dropdown, dataset_dropdown, extract_path_text

def extract_selected_dataset(category, dataset_file, extract_path):
    """Extract selected dataset"""
    
    # Build complete path
    zip_path = os.path.join("machinery-phm-tutorial/data_bases", category, dataset_file)
    
    if not os.path.exists(zip_path):
        print(f"❌ File does not exist: {zip_path}")
        return
    
    # Create extraction directory
    os.makedirs(extract_path, exist_ok=True)
    
    try:
        print(f"🔄 Extracting {dataset_file}...")
        
        # Extract main file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Check if there are nested zip files to extract
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.zip'):
                    nested_zip_path = os.path.join(root, file)
                    nested_extract_path = os.path.join(root, file.replace('.zip', ''))
                    
                    print(f"🔄 Found nested zip file, extracting: {file}")
                    os.makedirs(nested_extract_path, exist_ok=True)
                    
                    with zipfile.ZipFile(nested_zip_path, 'r') as nested_zip_ref:
                        nested_zip_ref.extractall(nested_extract_path)
        
        print(f"✅ Dataset extraction completed!")
        print(f"📁 Extract location: {extract_path}")
        
        # List extracted files
        print("\n📄 Extracted files:")
        for root, dirs, files in os.walk(extract_path):
            level = root.replace(extract_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}📁 {os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}📄 {file}")
        
        # If it's turbofan dataset, automatically load data
        if 'turbofan' in dataset_file.lower() or 'cmapss' in dataset_file.lower():
            load_turbofan_data(extract_path)
            
    except Exception as e:
        print(f"❌ Extraction failed: {str(e)}")

def load_turbofan_data(extract_path):
    """Load Turbofan dataset"""
    print("\n🔧 Loading Turbofan dataset...")
    
    # Find data files
    data_files = []
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.startswith('train_FD') and file.endswith('.txt'):
                data_files.append(os.path.join(root, file))
    
    if not data_files:
        print("❌ Turbofan training data files not found")
        return
    
    # Use the first found data file
    train_file = data_files[0]
    base_name = os.path.basename(train_file).replace('train_', '').replace('.txt', '')
    
    # Find corresponding test and RUL files
    train_dir = os.path.dirname(train_file)
    test_file = os.path.join(train_dir, f"test_{base_name}.txt")
    rul_file = os.path.join(train_dir, f"RUL_{base_name}.txt")
    
    # Column names definition
    columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
              [f'sensor_{i}' for i in range(1, 22)]
    
    try:
        # Load data
        train_df = pd.read_csv(train_file, sep=' ', header=None, names=columns)
        train_df = train_df.dropna(axis=1, how='all')
        
        print(f"✅ Training data loaded successfully: {train_df.shape}")
        
        if os.path.exists(test_file):
            test_df = pd.read_csv(test_file, sep=' ', header=None, names=columns)
            test_df = test_df.dropna(axis=1, how='all')
            print(f"✅ Test data loaded successfully: {test_df.shape}")
        
        if os.path.exists(rul_file):
            rul_df = pd.read_csv(rul_file, sep=' ', header=None, names=['RUL'])
            print(f"✅ RUL data loaded successfully: {rul_df.shape}")
        
        print("\n📊 Training data sample:")
        display(train_df.head())
        
        # Save data to global variables for later use
        globals()['train_df'] = train_df
        if 'test_df' in locals():
            globals()['test_df'] = test_df
        if 'rul_df' in locals():
            globals()['rul_df'] = rul_df
            
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")

# Create dataset selection interface
category_dropdown, dataset_dropdown, extract_path_text = create_dataset_selector()
