import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
import seaborn as sns

# === Paths ===
subsets = ['FD001', 'FD002', 'FD003', 'FD004']
columns = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
          [f'sensor_{i}' for i in range(1, 22)]
sensor_cols = [f'sensor_{i}' for i in range(1, 22)]

# Global variables to store data
global_df = None
data_root = None

# === Data root input widget ===
def create_data_root_input():
    style = {'description_width': '150px'}
    layout = widgets.Layout(width='500px')
    
    data_root_input = widgets.Text(
        value="/content/extracted_data/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData",
        placeholder="Enter path to CMAPSS data root directory",
        description='Data Root Path:',
        style=style,
        layout=layout
    )
    
    set_path_btn = widgets.Button(
        description='Set Data Path',
        button_style='primary',
        layout=widgets.Layout(width='150px')
    )
    
    output_area = widgets.Output()
    
    def on_set_path_clicked(b):
        global data_root
        with output_area:
            clear_output(wait=True)
            data_root = data_root_input.value
            print(f"✅ Data root path set to: {data_root}")
            if not os.path.exists(data_root):
                print("⚠️ Warning: Path does not exist!")
    
    set_path_btn.on_click(on_set_path_clicked)
    
    return widgets.VBox([
        widgets.HTML("<h3>📁 Data Path Configuration</h3>"),
        widgets.HBox([data_root_input, set_path_btn]),
        output_area
    ])

# === Load and preprocess function ===
def load_and_preprocess(subset):
    global data_root
    if data_root is None:
        raise ValueError("Data root path not set. Please set the data path first.")
    
    # Load train data
    train_path = os.path.join(data_root, f"train_{subset}.txt")
    
    df = pd.read_csv(train_path, sep='\s+', header=None, names=columns)
    
    # Compute RUL for each cycle
    max_cycles = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    df = df.merge(max_cycles, on='unit_number')
    df['RUL'] = df['max_cycle'] - df['time_in_cycles']
    df.drop('max_cycle', axis=1, inplace=True)
    
    return df

# === Simple visualization ===
def create_sensor_plot(df, sensor_names, unit_numbers, plot_output):
    """Create simple sensor visualization in the plot output widget"""
    
    n_sensors = len(sensor_names)
    n_cols = min(3, n_sensors)
    n_rows = (n_sensors + n_cols - 1) // n_cols
    
    with plot_output:
        clear_output(wait=True)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        
        if n_sensors == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Sensor Data Visualization', fontsize=16)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(unit_numbers)))
        
        for i, sensor in enumerate(sensor_names):
            ax = axes[i] if n_sensors > 1 else axes[0]
            
            for j, unit in enumerate(unit_numbers):
                unit_data = df[df['unit_number'] == unit]
                color = colors[j]
                
                ax.plot(unit_data['time_in_cycles'], unit_data[sensor], 
                       color=color, label=f'Unit {unit}', linewidth=1.5, alpha=0.8)
            
            ax.set_title(f'{sensor} Time Series')
            ax.set_xlabel('Time in Cycles')
            ax.set_ylabel(sensor)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_sensors, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()

# === Simple Configuration Widget ===
def create_simple_config_widget():
    global global_df
    
    # Style definition
    style = {'description_width': '150px'}
    layout = widgets.Layout(width='300px')
    
    # Widget components
    subset_selector = widgets.Dropdown(
        options=subsets,
        value='FD001',
        description='Select Subset:',
        style=style,
        layout=layout
    )
    
    sensor_selector = widgets.SelectMultiple(
        options=sensor_cols,
        value=['sensor_1'],
        description='Select Sensors:',
        style=style,
        layout=widgets.Layout(width='300px', height='120px')
    )
    
    unit_selector = widgets.SelectMultiple(
        options=[],
        value=[],
        description='Select Units:',
        style=style,
        layout=widgets.Layout(width='300px', height='120px')
    )
    
    # Buttons
    load_data_btn = widgets.Button(
        description='Load Data',
        button_style='primary',
        layout=widgets.Layout(width='150px')
    )
    
    plot_data_btn = widgets.Button(
        description='Plot Data',
        button_style='success',
        layout=widgets.Layout(width='150px')
    )
    
    # Output areas
    output_area = widgets.Output()
    plot_output = widgets.Output()
    
    # Event handlers
    def on_load_data_clicked(b):
        with output_area:
            clear_output(wait=True)
            if data_root is None:
                print("❌ Please set data root path first!")
                return
                
            print("Loading data...")
            
            try:
                global global_df
                global_df = load_and_preprocess(subset_selector.value)
                
                # Update unit selector options
                available_units = sorted(global_df['unit_number'].unique())
                unit_selector.options = available_units
                unit_selector.value = available_units[:min(5, len(available_units))]
                
                print(f"✅ Data loaded successfully!")
                print(f"   Dataset: {subset_selector.value}")
                print(f"   Total samples: {len(global_df)}")
                print(f"   Number of units: {len(available_units)}")
                
            except Exception as e:
                print(f"❌ Error loading data: {str(e)}")
    
    def on_plot_data_clicked(b):
        if global_df is None:
            with output_area:
                clear_output(wait=True)
                print("❌ Please load data first!")
            return
        
        if len(unit_selector.value) == 0:
            with output_area:
                clear_output(wait=True)
                print("❌ Please select at least one unit!")
            return
        
        if len(sensor_selector.value) == 0:
            with output_area:
                clear_output(wait=True)
                print("❌ Please select at least one sensor!")
            return
        
        try:
            selected_units = list(unit_selector.value)
            selected_sensors = list(sensor_selector.value)
            
            with output_area:
                clear_output(wait=True)
                print("🚀 Generating plot...")
            
            create_sensor_plot(global_df, selected_sensors, selected_units, plot_output)
                
        except Exception as e:
            with output_area:
                clear_output(wait=True)
                print(f"❌ Error plotting data: {str(e)}")
    
    # Connect event handlers
    load_data_btn.on_click(on_load_data_clicked)
    plot_data_btn.on_click(on_plot_data_clicked)
    
    # Left side control panel
    control_panel = widgets.VBox([
        widgets.HTML("<h3>📊 CMAPSS Data Visualization</h3>"),
        subset_selector,
        widgets.HBox([load_data_btn, plot_data_btn]),
        sensor_selector,
        unit_selector,
        output_area
    ], layout=widgets.Layout(width='400px', padding='10px'))
    
    # Right side plot area
    plot_panel = widgets.VBox([
        widgets.HTML("<h3>📈 Visualization Area</h3>"),
        plot_output
    ], layout=widgets.Layout(width='800px', padding='10px'))
    
    # Main layout - left controls, right plots
    main_layout = widgets.HBox([
        control_panel,
        plot_panel
    ])
    
    return main_layout

# === Main function ===
def raw_data_visualization():
    """Main function to initialize and display the CMAPSS raw data visualization interface"""
    print("🚀 Initializing Simple CMAPSS Data Visualization Tool")
    print("=" * 60)
    
    # Create and display data root input
    path_widget = create_data_root_input()
    display(path_widget)
    
    # Create and display main configuration widget
    config_widget = create_simple_config_widget()
    display(config_widget)
    
    print("\n📋 Instructions:")
    print("1. Set the data root path")
    print("2. Select a dataset subset")
    print("3. Load data")
    print("4. Select sensors and units to visualize")
    print("5. Plot the data")

if __name__ == "__main__":
    raw_data_visualization()
