import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.font_manager as fm
import ipywidgets as widgets
from IPython.display import display

def print_cmapss_sensor_descriptions():
    """Print the physical meanings of CMAPSS sensors"""
    print("CMAPSS (Commercial Modular Aero-Propulsion System Simulation) Sensor Descriptions")
    print("=" * 80)
    print()
    
    # Operational settings
    print("Operational Settings:")
    print("- setting1: Flight altitude (ft)")
    print("- setting2: Mach number")
    print("- setting3: Throttle resolver angle (degrees)")
    print()
    
    # Sensor descriptions
    sensor_descriptions = {
        'sensor_1': 'Total temperature at fan inlet (°R)',
        'sensor_2': 'Total temperature at LPC outlet (°R)',
        'sensor_3': 'Total temperature at HPC outlet (°R)',
        'sensor_4': 'Total temperature at LPT outlet (°R)',
        'sensor_5': 'Pressure at fan inlet (psia)',
        'sensor_6': 'Total pressure in bypass-duct (psia)',
        'sensor_7': 'Total pressure at HPC outlet (psia)',
        'sensor_8': 'Physical fan speed (rpm)',
        'sensor_9': 'Physical core speed (rpm)',
        'sensor_10': 'Engine pressure ratio (P50/P2)',
        'sensor_11': 'Static pressure at HPC outlet (psia)',
        'sensor_12': 'Ratio of fuel flow to Ps30 (pps/psi)',
        'sensor_13': 'Corrected fan speed (rpm)',
        'sensor_14': 'Corrected core speed (rpm)',
        'sensor_15': 'Bypass Ratio',
        'sensor_16': 'Burner fuel-air ratio',
        'sensor_17': 'Bleed Enthalpy',
        'sensor_18': 'Required fan speed (rpm)',
        'sensor_19': 'Required core speed (rpm)',
        'sensor_20': 'High-pressure turbine coolant bleed (lbm/s)',
        'sensor_21': 'Low-pressure turbine coolant bleed (lbm/s)'
    }
    
    print("Sensor Measurements:")
    for sensor, description in sensor_descriptions.items():
        print(f"- {sensor}: {description}")
    
    print()
    print("Component Abbreviations:")
    print("- LPC: Low Pressure Compressor")
    print("- HPC: High Pressure Compressor")
    print("- LPT: Low Pressure Turbine")
    print("- HPT: High Pressure Turbine")
    print()
    print("Key Degradation Indicators:")
    print("- Sensors 2, 3, 4: Temperature measurements (thermal efficiency)")
    print("- Sensors 7, 11: Pressure measurements (compression efficiency)")
    print("- Sensors 8, 9: Speed measurements (mechanical health)")
    print("- Sensors 13, 14: Corrected speeds (performance deterioration)")
    print("=" * 80)

def load_and_analyze_dataset(file_path, dataset_name):
    """Load and analyze a single dataset"""
    # Define column names and load the dataset
    columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor_{i}' for i in range(1, 22)]
    train_df = pd.read_csv(file_path, sep='\s+', header=None, names=columns)
    
    # Select only the first 50 cycles as healthy (pre-degradation) samples
    healthy_data = train_df[train_df['cycle'] <= 50].copy()
    
    # Cluster operating conditions using setting1, setting2, setting3
    features = healthy_data[['setting1', 'setting2', 'setting3']]
    healthy_data['op_mode'] = KMeans(n_clusters=6, random_state=42).fit_predict(features)
    
    return healthy_data, dataset_name

def plot_sensor_comparison(healthy_data1, dataset_name1, healthy_data2, dataset_name2):
    """Plot sensor 4 comparison between two datasets"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot first dataset
    sns.boxplot(
        data=healthy_data1,
        x='op_mode',
        y='sensor_4',
        hue='op_mode',
        palette='Set2',
        dodge=False,
        legend=False,
        ax=ax1
    )
    ax1.set_title(f"Sensor_4 (LPT Outlet Temperature) Across Operating Conditions ({dataset_name1})")
    ax1.set_xlabel("Operating Condition Cluster (KMeans)")
    ax1.set_ylabel("Sensor_4 Reading (°R)")
    ax1.grid(True)
    
    # Plot second dataset
    sns.boxplot(
        data=healthy_data2,
        x='op_mode',
        y='sensor_4',
        hue='op_mode',
        palette='Set3',
        dodge=False,
        legend=False,
        ax=ax2
    )
    ax2.set_title(f"Sensor_4 (LPT Outlet Temperature) Across Operating Conditions ({dataset_name2})")
    ax2.set_xlabel("Operating Condition Cluster (KMeans)")
    ax2.set_ylabel("Sensor_4 Reading (°R)")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def on_analyze_button_clicked(b):
    """Callback function for analyze button"""
    dataset1_path = f"/content/extracted_data/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData/{dataset1_dropdown.value}"
    dataset2_path = f"/content/extracted_data/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData/{dataset2_dropdown.value}"
    
    try:
        # Load and analyze both datasets
        healthy_data1, dataset_name1 = load_and_analyze_dataset(dataset1_path, dataset1_dropdown.value.replace('.txt', '').upper())
        healthy_data2, dataset_name2 = load_and_analyze_dataset(dataset2_path, dataset2_dropdown.value.replace('.txt', '').upper())
        
        # Plot comparison
        plot_sensor_comparison(healthy_data1, dataset_name1, healthy_data2, dataset_name2)
        
        print(f"Analysis completed for {dataset_name1} and {dataset_name2}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

def run_dataset_comparison_tool():
    """Main function to create UI and handle dataset comparison"""
    global dataset1_dropdown, dataset2_dropdown
    
    # Print sensor descriptions first
    print_cmapss_sensor_descriptions()
    print()
    
    # Available dataset options
    dataset_options = [
        'train_FD001.txt',
        'train_FD002.txt', 
        'train_FD003.txt',
        'train_FD004.txt'
    ]
    
    # Create dropdown widgets
    dataset1_dropdown = widgets.Dropdown(
        options=dataset_options,
        value='train_FD001.txt',
        description='Dataset 1:',
        style={'description_width': 'initial'}
    )
    
    dataset2_dropdown = widgets.Dropdown(
        options=dataset_options,
        value='train_FD002.txt',
        description='Dataset 2:',
        style={'description_width': 'initial'}
    )
    
    # Create analyze button
    analyze_button = widgets.Button(
        description='Analyze & Compare',
        button_style='success',
        layout=widgets.Layout(width='200px')
    )
    
    # Attach callback to button
    analyze_button.on_click(on_analyze_button_clicked)
    
    # Display UI
    print("Turbofan Engine Dataset Comparison Tool")
    print("=" * 50)
    display(widgets.VBox([
        widgets.HTML("<h3>Select two datasets to compare:</h3>"),
        dataset1_dropdown,
        dataset2_dropdown,
        widgets.HTML("<br>"),
        analyze_button
    ]))

if __name__ == "__main__":
    run_dataset_comparison_tool()
