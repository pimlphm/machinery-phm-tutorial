import os
import zipfile
import pandas as pd
import numpy as np
from IPython.display import display, clear_output
import ipywidgets as widgets

# === Configuration ===
DATA_BASES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data_bases')

# === Core Functions ===
def extract_selected_dataset(category: str, dataset_file: str, extract_path: str):
    """Extract a selected zip dataset and any nested zips."""
    zip_path = os.path.join(DATA_BASES_PATH, category, dataset_file)
    if not os.path.exists(zip_path):
        print(f"❌ File does not exist: {zip_path}")
        return

    os.makedirs(extract_path, exist_ok=True)
    print(f"🔄 Extracting {dataset_file}...")

    # Extract main zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Recursively unpack nested zips
    for root, _, files in os.walk(extract_path):
        for file in files:
            if file.endswith('.zip'):
                nested_zip = os.path.join(root, file)
                nested_dir = os.path.join(root, file[:-4])
                print(f"🔄 Found nested zip: {file}, extracting to {nested_dir}")
                os.makedirs(nested_dir, exist_ok=True)
                with zipfile.ZipFile(nested_zip, 'r') as nz:
                    nz.extractall(nested_dir)

    print(f"✅ Dataset extraction completed at '{extract_path}'")
    _list_extracted_files(extract_path)

    # Auto-load turbofan if detected
    if any(k in dataset_file.lower() for k in ('turbofan', 'cmapss')):
        load_turbofan_data(extract_path)


def _list_extracted_files(path: str):
    """Print directory tree of extracted files."""
    print("\n📄 Extracted files:")
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}📁 {os.path.basename(root)}/")
        for f in files:
            print(f"{indent}  📄 {f}")


def load_turbofan_data(extract_path: str):
    """Load CMAPSS/Turbofan train, test, and RUL files into pandas DataFrames."""
    print("\n🔧 Loading Turbofan dataset...")
    data_files = []
    for root, _, files in os.walk(extract_path):
        for fname in files:
            if fname.startswith('train_FD') and fname.endswith('.txt'):
                data_files.append(os.path.join(root, fname))

    if not data_files:
        print("❌ Turbofan training data files not found")
        return

    train_path = data_files[0]
    base = os.path.basename(train_path).replace('train_', '').replace('.txt', '')
    test_path = os.path.join(os.path.dirname(train_path), f"test_{base}.txt")
    rul_path = os.path.join(os.path.dirname(train_path), f"RUL_{base}.txt")

    cols = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor_{i}' for i in range(1,22)]
    train_df = pd.read_csv(train_path, sep=' ', header=None, names=cols).dropna(axis=1, how='all')
    print(f"✅ Training data loaded: {train_df.shape}")

    test_df = None
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path, sep=' ', header=None, names=cols).dropna(axis=1, how='all')
        print(f"✅ Test data loaded: {test_df.shape}")

    rul_df = None
    if os.path.exists(rul_path):
        rul_df = pd.read_csv(rul_path, sep=' ', header=None, names=['RUL'])
        print(f"✅ RUL data loaded: {rul_df.shape}")

    print("\n📊 Training data preview:")
    display(train_df.head())

    globals().update({
        'train_df': train_df,
        'test_df': test_df,
        'rul_df': rul_df
    })

# === Interactive Interface ===
def create_dataset_selector():
    """Display widget-based selection for data extraction."""
    cats = [d for d in ('prognostics', 'diagnostics') if os.path.exists(os.path.join(DATA_BASES_PATH, d))]
    if not cats:
        print("❌ No datasets found under data_bases/prognostics or diagnostics.")
        return None, None, None

    cat_dd = widgets.Dropdown(options=cats, description='Data Type:', style={'description_width':'initial'})
    ds_dd = widgets.Dropdown(options=[], description='Dataset:', style={'description_width':'initial'})
    path_txt = widgets.Text(value='extracted_data', description='Extract Path:', style={'description_width':'initial'})
    btn = widgets.Button(description='Confirm Selection', button_style='success', icon='check')
    out = widgets.Output()

    def update_ds(change):
        p = os.path.join(DATA_BASES_PATH, change['new'])
        zips = [f for f in os.listdir(p) if f.endswith('.zip')]
        ds_dd.options = zips
        ds_dd.value = zips[0] if zips else None

    cat_dd.observe(update_ds, names='value')
    update_ds({'new': cats[0]})

    def on_click(btn):
        with out:
            clear_output()
            if not cat_dd.value or not ds_dd.value:
                print("❌ Please select a valid category and dataset.")
                return
            extract_selected_dataset(cat_dd.value, ds_dd.value, path_txt.value)

    btn.on_click(on_click)
    print("🔧 Select dataset:")
    display(widgets.VBox([cat_dd, ds_dd, path_txt, btn, out]))
    return cat_dd, ds_dd, path_txt

# === Execution Guard ===
if __name__ == "__main__":
    create_dataset_selector()
