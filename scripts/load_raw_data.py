import os
import zipfile
import pandas as pd
import numpy as np
from IPython.display import display, clear_output, HTML, Image
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
        print_cmapss_dataset_info()
        load_turbofan_data(extract_path)


def print_cmapss_dataset_info():
    """Print comprehensive information about the C-MAPSS dataset with embedded visualization."""
    print("\n" + "="*80)
    print("🛩️  C-MAPSS DATASET OVERVIEW")
    print("="*80)
    
    # Dataset description
    print("\n📖 DATASET DESCRIPTION:")
    print("The C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset")
    print("consists of operational and sensor data from aircraft engines. It includes")
    print("features such as engine ID, operational cycle, operational settings, and")
    print("sensor measurements from 21 sensors.")
    print()
    
    # Display the turbofan engine diagram from image file
    image_path = os.path.join('machinery-phm-tutorial', 'Images', 'Prognostics', 'Tubofan', 'C_MPASS.png')
    if os.path.exists(image_path):
        print("🔧 Turbofan Engine Cross-Section:")
        display(Image(filename=image_path, width=500))
        print()
    else:
        # Fallback to embedded SVG if image not found
        turbofan_html = """
        <div style="text-align: center; margin: 20px 0;">
            <h3>🔧 Turbofan Engine Cross-Section</h3>
            <div style="border: 2px solid #ddd; border-radius: 10px; padding: 20px; background-color: #f9f9f9;">
                <svg width="500" height="300" viewBox="0 0 500 300" style="max-width: 100%; height: auto;">
                    <!-- Engine outline -->
                    <ellipse cx="250" cy="150" rx="200" ry="80" fill="#e6e6e6" stroke="#333" stroke-width="2"/>
                    
                    <!-- Fan section -->
                    <circle cx="100" cy="150" r="60" fill="#4CAF50" stroke="#333" stroke-width="2"/>
                    <text x="100" y="155" text-anchor="middle" font-size="12" font-weight="bold">FAN</text>
                    
                    <!-- LPC section -->
                    <rect x="160" y="110" width="60" height="80" fill="#2196F3" stroke="#333" stroke-width="2"/>
                    <text x="190" y="155" text-anchor="middle" font-size="10" font-weight="bold">LPC</text>
                    
                    <!-- HPC section -->
                    <rect x="220" y="120" width="40" height="60" fill="#FF9800" stroke="#333" stroke-width="2"/>
                    <text x="240" y="155" text-anchor="middle" font-size="10" font-weight="bold">HPC</text>
                    
                    <!-- Combustor -->
                    <rect x="260" y="125" width="30" height="50" fill="#F44336" stroke="#333" stroke-width="2"/>
                    <text x="275" y="155" text-anchor="middle" font-size="9" font-weight="bold">CC</text>
                    
                    <!-- HPT section -->
                    <rect x="290" y="120" width="40" height="60" fill="#9C27B0" stroke="#333" stroke-width="2"/>
                    <text x="310" y="155" text-anchor="middle" font-size="10" font-weight="bold">HPT</text>
                    
                    <!-- LPT section -->
                    <rect x="330" y="110" width="60" height="80" fill="#607D8B" stroke="#333" stroke-width="2"/>
                    <text x="360" y="155" text-anchor="middle" font-size="10" font-weight="bold">LPT</text>
                    
                    <!-- Nozzle -->
                    <polygon points="390,130 450,140 450,160 390,170" fill="#795548" stroke="#333" stroke-width="2"/>
                    <text x="420" y="155" text-anchor="middle" font-size="9" font-weight="bold">NOZ</text>
                    
                    <!-- Airflow arrows -->
                    <defs>
                        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
                        </marker>
                    </defs>
                    <line x1="20" y1="150" x2="80" y2="150" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
                    <line x1="470" y1="150" x2="490" y2="150" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
                    
                    <!-- Labels -->
                    <text x="50" y="140" text-anchor="middle" font-size="10">Air In</text>
                    <text x="480" y="140" text-anchor="middle" font-size="10">Exhaust</text>
                </svg>
            </div>
        </div>
        """
        
        display(HTML(turbofan_html))
    
    # Sample data structure
    print("📊 SAMPLE DATA STRUCTURE:")
    sample_data_html = """
    <div style="margin: 15px 0;">
        <table style="border-collapse: collapse; width: 100%; max-width: 800px; margin: 0 auto;">
            <thead style="background-color: #f0f0f0;">
                <tr>
                    <th style="border: 1px solid #ddd; padding: 8px;">engine_id</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">cycle</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">setting1</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">setting2</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">sensor_1</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">sensor_2</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">...</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">1</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">1</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.8400</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">100.0</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">555.32</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">555.32</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">...</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">1</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">2</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.8408</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">100.0</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">462.54</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">1597.31</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">...</td>
                </tr>
            </tbody>
        </table>
    </div>
    """
    
    display(HTML(sample_data_html))
    
    print("\n🔧 KEY COMPONENTS:")
    print("• FAN: Air intake and bypass flow generation")
    print("• LPC: Low Pressure Compressor - initial air compression") 
    print("• HPC: High Pressure Compressor - high-pressure air generation")
    print("• CC: Combustion Chamber - fuel burning and energy generation")
    print("• HPT: High Pressure Turbine - drives HPC")
    print("• LPT: Low Pressure Turbine - drives LPC and fan")
    print("• NOZ: Nozzle - exhaust acceleration")
    
    print("\n📈 DATASET VARIANTS:")
    print("• FD001: Single fault mode, single operating condition")
    print("• FD002: Single fault mode, six operating conditions") 
    print("• FD003: Two fault modes, single operating condition")
    print("• FD004: Two fault modes, six operating conditions")
    
    print("\n🎯 APPLICATIONS:")
    print("• Remaining Useful Life (RUL) prediction")
    print("• Fault detection and diagnostics")
    print("• Condition monitoring and health assessment")
    print("• Predictive maintenance strategies")
    
    print("="*80)


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
